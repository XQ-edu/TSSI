import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Construct


def data_provider(args, flag, ratio):
    # ratio: a int number or a list of int numbers
    timeenc = 0 if args.embed != "timeF" else 1
    getitem_type = (
        0 if args.data == "ETTm1" or args.data == "ETTm2" or args.data == "weather" else 1
    )  # 0 for instantaneous value data, such as ETTm1, ETTm2, weather; 1 for cumulative value data, such as electricity, traffic
    if flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    data_set = Dataset_Construct(
        resample_data_dir=args.resample_data_dir,
        ratio=ratio,
        data_name=args.data,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        getitem_type=getitem_type,
    )
    if type(ratio) == int:
        print(f"ratio={ratio}, {flag}, instances: {len(data_set)}")
    else:
        print(f"all ratios, {flag}, instances: {len(data_set)}")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
