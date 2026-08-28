import math
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_frequency_repr(x, eps=1e-10):
    freq_repr = torch.abs(torch.fft.rfft(x, dim=-1, norm="forward"))
    freq_repr = torch.log1p(freq_repr)
    return F.normalize(freq_repr, p=2, dim=-1, eps=eps)


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class LocalMask:
    def __init__(self, B, L, S, device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)
            self._mask2 = ~torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=-self.len
            ).to(device)
            self._mask = self._mask1 + self._mask2

    @property
    def mask(self):
        return self._mask


class LearnedMahalanobisMask(nn.Module):
    """Build TSSI's fixed binary variable-relation mask."""

    def __init__(self, seq_len, low_pass_len, eps=1e-10):
        super().__init__()
        if isinstance(low_pass_len, bool) or not isinstance(low_pass_len, int):
            raise ValueError("low_pass_len must be an int")
        if low_pass_len < 1:
            raise ValueError("low_pass_len must be >= 1")

        seq_len = [seq_len] if isinstance(seq_len, int) else list(seq_len)
        if not seq_len:
            raise ValueError("seq_len must contain at least one branch")
        if any(low_pass_len > length // 2 + 1 for length in seq_len):
            raise ValueError(
                "low_pass_len must not exceed any branch's frequency length"
            )

        self.eps = eps
        self.low_pass_len = low_pass_len
        self.metric = nn.Parameter(torch.empty(low_pass_len, low_pass_len))
        nn.init.xavier_uniform_(self.metric)

    def _pairwise_distance(self, x1, x2):
        x1 = x1[..., : self.low_pass_len]
        x2 = x2[..., : self.low_pass_len]
        metric = self.metric.to(device=x1.device, dtype=x1.dtype)

        diff = x1.unsqueeze(2) - x2.unsqueeze(1)
        projected = torch.einsum("dk,bxck->bxcd", metric, diff)
        squared_distance = torch.einsum("bxcd,bxcd->bxc", projected, projected)
        return torch.log(squared_distance + self.eps)

    def _distance_matrix(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        elif isinstance(x, (list, tuple)):
            x = list(x)
        else:
            raise TypeError(
                "x must be a tensor [B, N, L] or a list of such tensors"
            )

        freq = [to_frequency_repr(branch, eps=self.eps) for branch in x]
        branch_count = len(freq)
        blocks = cast(
            list[list[torch.Tensor]],
            [[None for _ in range(branch_count)] for _ in range(branch_count)],
        )
        for i in range(branch_count):
            for j in range(i, branch_count):
                distance = self._pairwise_distance(freq[i], freq[j])
                blocks[i][j] = distance
                blocks[j][i] = distance if i == j else distance.transpose(1, 2)

        return torch.cat(
            [torch.cat(blocks[i], dim=-1) for i in range(branch_count)], dim=1
        )

    def _sample_training_mask(self, logits):
        uniform = torch.rand_like(logits).clamp(self.eps, 1.0 - self.eps)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        soft_mask = torch.sigmoid(logits + logistic_noise)
        hard_mask = (soft_mask > 0.5).to(soft_mask.dtype)
        return hard_mask.detach() - soft_mask.detach() + soft_mask

    def forward(self, x, distance_threshold):
        distance = self._distance_matrix(x)
        channel_count = distance.shape[-1]
        if distance_threshold.numel() != channel_count:
            raise ValueError(
                "distance_threshold must have one value per input channel"
            )

        threshold = distance_threshold.to(
            device=distance.device, dtype=distance.dtype
        ).reshape(1, channel_count, 1)
        logits = threshold - distance
        diagonal = torch.eye(
            channel_count, device=logits.device, dtype=torch.bool
        ).unsqueeze(0)
        sample_logits = logits.masked_fill(diagonal, 0.0)

        if self.training:
            sample = self._sample_training_mask(sample_logits)
        else:
            sample = (sample_logits > 0).to(logits.dtype)

        identity = diagonal.to(sample.dtype)
        sample = sample * (1.0 - identity) + identity
        return sample.unsqueeze(1)
