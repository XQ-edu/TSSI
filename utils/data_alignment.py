import numpy as np
import torch
import torch.nn.functional as F


def split(data, ratio):
    data = data / ratio
    return torch.repeat_interleave(data, ratio, dim=1)

def repeat(data, ratio):
    return torch.repeat_interleave(data, ratio, dim=1)

def zero(data, ratio):
    B, L, D = data.shape
    new_L = L * ratio
    result = torch.zeros(B, new_L, D, device=data.device)
    # result[:, :L, :] = data
    # Place original data points and pad zeros after each value
    for i in range(L):
        result[:, i * ratio, :] = data[:, i, :]
        # The remaining (ratio-1) positions after each original value remain zeros
    return result


def linear_right(data, ratio):
    """
    Block-wise linear interpolation. For each original time point,
    insert (ratio-1) interpolated points after the current original point.
    """
    B, L, D = data.shape
    new_L = L * ratio
    result = torch.zeros(B, new_L, D, device=data.device)

    # Original implementation: original points at block start
    for i in range(L):
        start_idx = i * ratio
        # Place original data point at the beginning of each block
        result[:, start_idx, :] = data[:, i, :]

        # Interpolate within each block (except for the last block)
        if i < L - 1:
            # Linear interpolation between current and next point
            for j in range(1, ratio):
                alpha = j / ratio  # interpolation weight
                result[:, start_idx + j, :] = (1 - alpha) * data[
                    :, i, :
                ] + alpha * data[:, i + 1, :]
        else:
            # For the last block, extrapolate using linear trend
            if L >= 2:
                # Calculate the trend from the last two points
                trend = data[:, i, :] - data[:, i - 1, :]
                for j in range(1, ratio):
                    # Extrapolate: last_point + j * trend_per_step
                    result[:, start_idx + j, :] = data[:, i, :] + j * (trend / ratio)
            else:
                # If only one point, just repeat it
                for j in range(1, ratio):
                    result[:, start_idx + j, :] = data[:, i, :]
    return result


def linear_left(data, ratio):
    """
    Block-wise linear interpolation. For each original time point,
    insert (ratio-1) interpolated points before the current original point.
    """
    B, L, D = data.shape
    new_L = L * ratio
    result = torch.zeros(B, new_L, D, device=data.device)

    # New implementation: original points at block end
    for i in range(L):
        end_idx = (i + 1) * ratio - 1  # End position of current block
        # Place original data point at the end of each block
        result[:, end_idx, :] = data[:, i, :]

        # Interpolate within each block (except for the first block)
        if i > 0:
            # Linear interpolation between previous and current point
            for j in range(ratio - 1):
                alpha = (j + 1) / ratio  # interpolation weight
                result[:, i * ratio + j, :] = (1 - alpha) * data[
                    :, i - 1, :
                ] + alpha * data[:, i, :]
        else:
            # For the first block, extrapolate using linear trend (if possible)
            if L >= 2:
                # Calculate the trend from the first two points
                trend = data[:, 1, :] - data[:, 0, :]
                for j in range(ratio - 1):
                    # Extrapolate backward: first_point - (ratio-1-j) * trend_per_step
                    result[:, j, :] = data[:, 0, :] - (ratio - 1 - j) * (trend / ratio)
            else:
                # If only one point, just repeat it
                for j in range(ratio - 1):
                    result[:, j, :] = data[:, 0, :]
    return result



def data_alignment(data_list, ratio_list, method="repeat"):
    """
    Align the length of data in data_list.
    Args:
        data_list: list of torch tensor, each element is [batch_size, seq_len[i], corresponding enc_in]
        ratio_list: a list of int numbers, the data sampling ratio = highest / lower
        method: str, the method to align the length of data, options:
                - "repeat": repeat each time point ratio times
                - "split": divide values and repeat (for traffic/electricity data)
                - "zero": pad with zeros after original data
                - "linear_right": interpolate with original points at block start
                - "linear_left": interpolate with original points at block end
    Returns:
        align_data: torch tensor, [batch_size, aligned_seq_len, total_enc_in]
    """

    method_dict = {
        "split": split,
        "repeat": repeat,
        "zero": zero,
        "linear_right": linear_right,
        "linear_left": linear_left,
    }

    align_data = []
    for data, ratio in zip(data_list, ratio_list):
        align_data.append(method_dict[method](data, ratio) if ratio != 1 else data)
    align_data = torch.cat(align_data, dim=-1)
    return align_data
