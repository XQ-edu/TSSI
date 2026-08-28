import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import LearnedMahalanobisMask


class ResidualMLPLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super(ResidualMLPLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, in_dim)
        return x + self.fc2(self.dropout(self.act(self.fc1(x))))  # (B, N, in_dim)


class CITFHead(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        time_feat_dim,
        in_dim,
        hidden_dim,
        n_res_layers,
        dropout=0.1,
    ):
        super(CITFHead, self).__init__()
        self.value_emb = nn.Linear(seq_len, in_dim)
        self.x_time_emb = nn.Linear(time_feat_dim * seq_len, in_dim)

        self.encoder = nn.Sequential(
            *[ResidualMLPLayer(in_dim, hidden_dim, dropout) for _ in range(n_res_layers)]
        )

        self.pred = nn.Linear(in_dim, pred_len)

    def forward(self, x, x_time):
        """
        input:
            x: (B, N, seq_len)
        x_time: (B, seq_len, time_feat_dim)
        output:
            out: (B, N, pred_len)
        """
        B, N, _ = x.shape

        x_emb = self.value_emb(x)  # (B, N, in_dim)

        x_time_emb = self.x_time_emb(x_time.reshape(B, -1))  # (B, in_dim)
        x_time_emb = x_time_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, in_dim)
        x_emb = x_emb + x_time_emb

        out = self.encoder(x_emb)  # (B, N, in_dim)
        out = self.pred(out)  # (B, N, pred_len)
        return out


class GlobalToken(nn.Module):
    def __init__(self, enc_in, d_model, token_num=1):
        super(GlobalToken, self).__init__()
        self.global_token = nn.Parameter(torch.randn(1, enc_in, token_num, d_model))

    def forward(self, x):
        """
        input:
            x: [B, N, seq_len]
        output:
            out: [B, N, token_num, d_model]
        """
        B = x.shape[0]
        global_token = self.global_token.expand(B, -1, -1, -1)
        return global_token


class AllTokens(nn.Module):
    def __init__(
        self,
        ratio_list,
        enc_in,
        d_model,
        seq_len,
        recent_factor=1,
    ):
        super(AllTokens, self).__init__()
        self.ratio_num = len(ratio_list)
        self.recent_len = [
            recent_factor * (max(ratio_list) // ratio_list[i])
            for i in range(self.ratio_num)
        ]
        if any(
            seq_len[i] <= self.recent_len[i] for i in range(self.ratio_num)
        ):
            raise ValueError("seq_len must be > recent_len for every ratio branch")
        self.distant_token = nn.ModuleList(
            [
                nn.Linear(
                    seq_len[i] - self.recent_len[i],
                    d_model,
                )
                for i in range(self.ratio_num)
            ]
        )
        self.token_pos_emb = nn.ModuleList(
            [PositionalEmbedding(d_model) for _ in range(self.ratio_num)]
        )
        self.recent_token = nn.ModuleList(
            [
                nn.Linear(
                    self.recent_len[i],
                    d_model,
                )
                for i in range(self.ratio_num)
            ]
        )
        self.global_token = nn.ModuleList(
            [
                GlobalToken(
                    enc_in=enc_in[i],
                    d_model=d_model,
                )
                for i in range(self.ratio_num)
            ]
        )

    def add_token_pos_emb(self, token, idx, pos_offset=0):
        pos_embs = self.token_pos_emb[idx].pe[
            :, pos_offset : pos_offset + token.shape[2]
        ]
        pos_embs = pos_embs.unsqueeze(1).expand(-1, token.shape[1], -1, -1)
        return token + pos_embs

    def forward(self, x):
        distant_x = [
            x[i][:, :, : -self.recent_len[i]] for i in range(self.ratio_num)
        ]  # list of [B, N, seq_len[i] - recent_len[i]]

        recent_x = [
            x[i][:, :, -self.recent_len[i] :] for i in range(self.ratio_num)
        ]  # list of [B, N, recent_len]

        distant_token = [
            self.add_token_pos_emb(
                self.distant_token[i](distant_x[i]).unsqueeze(2), i, 0
            )
            for i in range(self.ratio_num)
        ]  # list of [B, N, 1, d_model]

        recent_token = [
            self.add_token_pos_emb(
                self.recent_token[i](recent_x[i]).unsqueeze(2),
                i,
                1,
            )
            for i in range(self.ratio_num)
        ]  # list of [B, N, 1, d_model]

        global_token = [
            self.global_token[i](x[i]) for i in range(self.ratio_num)
        ]  # list of [B, N, 1, d_model]

        return distant_token, recent_token, global_token


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(2)]


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(
        self,
        x,
        attn_bias=None,
        attn_mask=None,
        tau=None,
        delta=None,
    ):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(
                    x,
                    attn_bias=attn_bias,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=delta,
                )
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](
                x,
                attn_bias=attn_bias,
                attn_mask=attn_mask,
                tau=tau,
                delta=delta,
            )
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x,
                    attn_bias=attn_bias,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=delta,
                )
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x,
        attn_bias=None,
        attn_mask=None,
        tau=None,
        delta=None,
    ):
        new_queries, attn = self.attention(
            x,
            x,
            x,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )
        x = x + self.dropout(new_queries)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class AttentionLayer(nn.Module):
    def __init__(
        self, attention, d_model, n_heads, d_keys=None, d_values=None, gate_type=None
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

        self.gate_type = gate_type
        if self.gate_type == "elementwise":
            self.query_projection = nn.Linear(d_model, d_keys * n_heads * 2)
        elif self.gate_type == "headwise":
            self.query_projection = nn.Linear(d_model, d_keys * n_heads + n_heads)
        else:
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)

    def forward(
        self,
        queries,
        keys,
        values,
        attn_bias=None,
        attn_mask=None,
        tau=None,
        delta=None,
    ):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        gate = None
        queries_proj = self.query_projection(queries)
        if self.gate_type == "elementwise":
            queries_proj = queries_proj.view(B, L, H, -1)
            queries, gate = torch.split(
                queries_proj, [self.d_keys, self.d_keys], dim=-1
            )
        elif self.gate_type == "headwise":
            queries_proj = queries_proj.view(B, L, H, -1)
            queries, gate = torch.split(queries_proj, [self.d_keys, 1], dim=-1)
        else:
            queries = queries_proj.view(B, L, H, -1)

        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_bias,
            attn_mask,
        )
        if gate is not None:
            out = out * torch.sigmoid(gate)

        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries,
        keys,
        values,
        attn_bias=None,
        attn_mask=None,
        tau=None,
        delta=None,
    ):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scores = scores * scale

        if attn_bias is not None:
            if attn_bias.dtype == torch.bool:
                if attn_bias.dim() == 3:
                    attn_bias = attn_bias.unsqueeze(1)
                elif attn_bias.dim() != 4:
                    raise ValueError(
                        f"Unsupported bool attn_bias shape: {attn_bias.shape}"
                    )
                scores = scores.masked_fill(~attn_bias, -np.inf)
            else:
                if attn_bias.dim() == 3:
                    attn_bias = attn_bias.unsqueeze(1)
                elif attn_bias.dim() != 4:
                    raise ValueError(
                        f"Unsupported float attn_bias shape: {attn_bias.shape}"
                    )
                scores = scores + attn_bias

        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)
            scores = scores * attn_mask + attention_mask

        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.ratio_list = configs.ratio
        self.ratio_num = len(self.ratio_list)
        self.use_norm = configs.use_norm

        self.seq_len = [configs.seq_len // r for r in self.ratio_list]
        self.pred_len = [configs.pred_len // r for r in self.ratio_list]
        self.enc_total = configs.enc_in
        self.enc_in = []
        self.enc_idx = [0]
        for i in range(self.ratio_num):
            if i < self.ratio_num - 1:
                self.enc_in.append(configs.enc_in // self.ratio_num)
            else:
                self.enc_in.append(
                    configs.enc_in // self.ratio_num + configs.enc_in % self.ratio_num
                )
            self.enc_idx.append(sum(self.enc_in))
        self.recent_factor = getattr(configs, "recent_factor", 2)
        if isinstance(self.recent_factor, bool) or not isinstance(
            self.recent_factor, int
        ):
            raise ValueError("recent_factor must be an int")
        if self.recent_factor < 1:
            raise ValueError("recent_factor must be >= 1")

        # Fixed TSSI mechanism parameters.
        self.time_feat_dim = 1
        if configs.freq == "h":
            self.time_feat_dim = 4
        elif configs.freq == "t":
            self.time_feat_dim = 5
        self.low_pass_len = min(self.seq_len) // 2
        if self.low_pass_len < 1:
            raise ValueError("min(seq_len branches) // 2 must be >= 1")
        self.mask_distance_threshold = nn.Parameter(
            torch.full((self.enc_total,), -1.0)
        )

        print(
            "Model self params:",
            {
                "ratio_list": self.ratio_list,
                "ratio_num": self.ratio_num,
                "use_norm": self.use_norm,
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "enc_total": self.enc_total,
                "enc_in": self.enc_in,
                "enc_idx": self.enc_idx,
                "time_feat_dim": self.time_feat_dim,
                "recent_factor": self.recent_factor,
                "low_pass_len": self.low_pass_len,
            },
        )

        # ---------------- CITF branch ----------------
        self.citf = nn.ModuleList(
            [
                CITFHead(
                    seq_len=self.seq_len[i],
                    pred_len=self.pred_len[i],
                    time_feat_dim=self.time_feat_dim,
                    in_dim=configs.in_dim,
                    hidden_dim=configs.hidden_dim,
                    n_res_layers=configs.n_res_layers,
                    dropout=configs.dropout,
                )
                for i in range(self.ratio_num)
            ]
        )
        # ----------------- all tokens ----------------
        self.all_tokens = AllTokens(
            ratio_list=self.ratio_list,
            enc_in=self.enc_in,
            d_model=configs.d_model,
            seq_len=self.seq_len,
            recent_factor=self.recent_factor,
        )
        # ----------------- temporal encoder ----------------
        self.temporal_encoder = nn.ModuleList(
            [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(
                                    False,
                                    configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention,
                                ),
                                configs.d_model,
                                configs.n_heads,
                                gate_type="elementwise",
                            ),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                        )
                        for _ in range(configs.e_layers)
                    ],
                    norm_layer=nn.LayerNorm(configs.d_model),
                )
                for i in range(self.ratio_num)
            ]
        )

        # ---------------- channel encoder ----------------
        self.mask_generator = LearnedMahalanobisMask(
            seq_len=self.seq_len,
            low_pass_len=self.low_pass_len,
        )
        self.channel_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                        gate_type="elementwise",
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # ---------------- RSCI branch ----------------
        rsci_input_dim = configs.d_model * 3
        self.rsci = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(rsci_input_dim),
                    nn.Linear(
                        rsci_input_dim,
                        self.pred_len[i],
                    ),
                )
                for i in range(self.ratio_num)
            ]
        )
        # adaptive fusion parameters
        # each channel has a learnable weight to balance CITF and RSCI predictions
        self.alpha_logits = nn.Parameter(torch.zeros(configs.enc_in))

    def forecast(self, x, x_time):
        """
        x: list, each element is [B, seq_len[i], enc_in[i]]
        x_time: list, each element is [B, seq_len[i], time_feat_dim]
        """
        x = [v.permute(0, 2, 1) for v in x]  # list of [B, enc_in[i], seq_len[i]]
        B = x[0].shape[0]
        channel_mask = self.mask_generator(
            x,
            distance_threshold=self.mask_distance_threshold,
        )  # [B, 1, enc_total, enc_total]

        # ---------------- forecasting only by CITF information ----------------
        citf_preds = [
            pred(x[i], x_time[i])
            for i, pred in enumerate(self.citf)
        ]  # list of [B, enc_in[i], pred_len[i]]
        distant_token, recent_token, global_token = self.all_tokens(x)

        temp_out = []
        for i in range(self.ratio_num):
            temp_in = torch.cat(
                [distant_token[i], recent_token[i], global_token[i]], dim=2
            ).reshape(
                B * self.enc_in[i],
                -1,
                distant_token[i].shape[-1],
            )
            out, _ = self.temporal_encoder[i](
                temp_in,
                attn_bias=None,
                attn_mask=None,
            )
            out = out.reshape(B, self.enc_in[i], -1, out.shape[-1])
            distant_token[i] = out[:, :, :1, :].reshape(
                B, self.enc_in[i], -1
            )
            global_token[i] = out[:, :, -1:, :].reshape(
                B, self.enc_in[i], -1
            )
            temp_out.append(out[:, :, 1:2, :])  # [B, enc_in[i], 1, d_model]
        chan_in = torch.cat(temp_out, dim=1).squeeze(2)

        chan_out, _ = self.channel_encoder(
            chan_in,
            attn_bias=None,
            attn_mask=channel_mask,
        )

        chan_out = torch.split(
            chan_out, self.enc_in, dim=1
        )  # list of [B, enc_in[i], d_model]

        chan_out = [
            torch.cat(
                [
                    distant_token[i],
                    chan_out[i],
                    global_token[i],
                ],
                dim=2,
            )
            for i in range(self.ratio_num)
        ]

        rsci_preds = [
            self.rsci[i](chan_out[i]) for i in range(self.ratio_num)
        ]  # list of [B, enc_in[i], pred_len[i]]

        # Combine CITF and RSCI predictions.
        alpha = torch.sigmoid(self.alpha_logits)
        chan_preds = [
            (
                rsci_preds[i]
                * alpha[self.enc_idx[i] : self.enc_idx[i + 1]].view(1, -1, 1)
                + (1 - alpha[self.enc_idx[i] : self.enc_idx[i + 1]].view(1, -1, 1))
                * citf_preds[i]
            )
            for i in range(self.ratio_num)
        ]
        return chan_preds

    def forward(self, x, x_time):
        """
        input:
            x: list, each element is [B, seq_len[i], enc_in[i]]
            x_time: list, each element is [B, seq_len[i], time_feat_dim]
        output:
            out: list, each element is [B, pred_len[i], enc_in[i]]
        """
        means = None
        stdevs = None
        if self.use_norm:
            # Normalization from Non-stationary Transformer for each channel group with different ratio
            means = [x[i].mean(1, keepdim=True).detach() for i in range(self.ratio_num)]
            stdevs = [
                torch.sqrt(torch.var(x[i], dim=1, keepdim=True, unbiased=False) + 1e-5)
                for i in range(self.ratio_num)
            ]
            x = [(x[i] - means[i]) / stdevs[i] for i in range(self.ratio_num)]
        dec_out = self.forecast(x, x_time)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer for each channel group with different ratio
            assert means is not None and stdevs is not None
            dec_out = [
                dec_out[i].permute(0, 2, 1) * stdevs[i] + means[i]
                for i in range(self.ratio_num)
            ]
        else:
            dec_out = [dec_out[i].permute(0, 2, 1) for i in range(self.ratio_num)]

        return dec_out
