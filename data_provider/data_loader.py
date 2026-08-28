import os
import warnings
from math import floor

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


class Dataset_Construct(Dataset):
    def __init__(
        self,
        resample_data_dir,
        ratio,
        data_name,
        flag="train",
        size=None,
        scale=True,
        timeenc=0,
        freq="h",
        getitem_type=0,
    ):
        """
        size: list, [seq_len, lable_len, pred_len], these lengths are setted based on highest
        ratio: a int number or a list of int numbers, the data sampling ratio = highest / lower
        enc_in: int, the variate number
        getitem_type: 0 or 1. 0 for instantaneous value data, such as ETTm1, ETTm2, weather; 1 for cumulative value data, such as electricity, traffic
        """
        # info
        # hightest rate corresponding length
        self.seq_len = size[0]
        self.lable_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "vali"]
        self.flag = flag

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.resample_data_dir = resample_data_dir
        self.data_name = data_name
        self.ratio = ratio
        self.getitem_type = getitem_type
        self.__read_data__()

    def read_single_rate_data(self, ratio, scaler):
        data_path = os.path.join(self.resample_data_dir, f"{ratio}", f"{self.flag}.csv")
        df_raw = pd.read_csv(data_path)

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_raw = pd.read_csv(
                os.path.join(self.resample_data_dir, f"{ratio}", "train.csv")
            )
            cols_data = train_raw.columns[1:]
            train_data = train_raw[cols_data]
            scaler.fit(train_data.values)
            data_xy = scaler.transform(df_data.values)
            # Access mean and std
            scaler_mean = scaler.mean_  # [var_num]
            scaler_std = scaler.scale_  # [var_num]
            # 保存均值和标准差到对应 ratio 目录下（单个 npz 文件），已存在则不重复保存
            save_dir = os.path.join(self.resample_data_dir, f"{ratio}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "scaler_stats.npz")
            if not os.path.exists(save_path):
                np.savez_compressed(save_path, mean=scaler_mean, std=scaler_std)
        else:
            data_xy = df_data.values

        df_stamp = df_raw[["date"]]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            # if freq is 'h', the time features are [hourofday, dayofweek, dayofmonth, dayofyear]
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        return data_xy, data_stamp

    def __read_data__(self):
        if type(self.ratio) == int:
            self.scaler = StandardScaler()
            self.data_xy, self.data_stamp = self.read_single_rate_data(
                self.ratio, self.scaler
            )
        else:
            self.data_xy = []
            self.data_stamp = []
            self.scaler = [None] * len(self.ratio)
            for i, ratio in enumerate(self.ratio):
                self.scaler[i] = StandardScaler()
                data_xy, data_stamp = self.read_single_rate_data(ratio, self.scaler[i])
                self.data_xy.append(data_xy)
                self.data_stamp.append(data_stamp)

    def __getitem__(self, index):
        if type(self.ratio) == int:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.lable_len
            r_end = r_begin + self.lable_len + self.pred_len

            seq_x = self.data_xy[s_begin:s_end]
            seq_y = self.data_xy[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            seq_x = []
            seq_y = []
            seq_x_mark = []
            seq_y_mark = []
            instance_idx = []
            time_offset = []

            for i, ratio in enumerate(self.ratio):
                seq_len = self.seq_len // ratio
                pred_len = self.pred_len // ratio
                lable_len = self.lable_len // ratio

                if (
                    self.getitem_type == 0
                ):  # instantaneous value data, such as ETTm1, ETTm2
                    s_begin = (index - 1) // ratio + 1
                    time_offset.append((index - 1) % ratio)
                else:  # cumulative value data, such as electricity, traffic
                    s_begin = index // ratio
                    time_offset.append(index % ratio)
                s_end = s_begin + seq_len
                r_begin = s_end - lable_len
                r_end = r_begin + lable_len + pred_len

                # each rate variables have their own seq_len and pred_len
                seq_x.append(self.data_xy[i][s_begin:s_end])
                seq_y.append(self.data_xy[i][r_begin:r_end])
                seq_x_mark.append(self.data_stamp[i][s_begin:s_end])
                seq_y_mark.append(self.data_stamp[i][r_begin:r_end])

                # whether the instance is reused, label the instance with unique index
                instance_idx.append(s_begin)
                # if index == 0 or (index - 1) % ratio == 0:
                #     # not reused
                #     seq_reuse.append(1)
                # else:  # reused
                #     seq_reuse.append(0)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, instance_idx, time_offset

    def __len__(self):
        if type(self.ratio) == int:
            return len(self.data_xy) - self.seq_len - self.pred_len + 1
        else:
            max_len = 0
            for i in range(len(self.ratio)):
                max_len = max(max_len, len(self.data_xy[i]))
            return max_len - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if type(self.ratio) == int:
            return self.scaler.inverse_transform(data)
        else:
            inverse_data = []
            for i in range(len(self.ratio)):
                inverse_data.append(self.scaler[i].inverse_transform(data[i]))
            return inverse_data
