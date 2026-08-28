import argparse
import ast
import gc
import os
import random


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: true, false, 1, 0")


fix_seed = 2025
random.seed(fix_seed)

parser = argparse.ArgumentParser(
    description="Autoformer & Transformer family for Time Series Forecasting"
)

# basic config
parser.add_argument("--is_training", type=int, default=1, help="status")
parser.add_argument(
    "--model",
    type=str,
    default="DLinear",
    help="registered forecasting model name",
)

# data loader
parser.add_argument("--data", type=str, default="ETTh2", help="dataset type")
parser.add_argument(
    "--root_path",
    type=str,
    default="./datasets/ETT-small",
    help="root path of the data file",
)
parser.add_argument("--data_path", type=str, default="ETTm1.csv", help="data file")
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)
parser.add_argument(
    "--results",
    type=str,
    default="./results/",
    help="location of model prediction results",
)
parser.add_argument(
    "--save_result",
    type=str_to_bool,
    default=True,
    help="whether to save prediction arrays and metrics under --results",
)

# target model settings
parser.add_argument(
    "--fc_dropout", type=float, default=0.05, help="fully connected dropout"
)
parser.add_argument("--patch_len", type=int, default=16, help="patch length")

# forecasting task
parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument("--label_len", type=int, default=48, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)

# DLinear
parser.add_argument(
    "--individual",
    action="store_true",
    default=False,
    help="DLinear: a linear layer for each variate(channel) individually",
)

# shared model settings
parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
parser.add_argument("--c_out", type=int, default=7, help="output size")
parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
parser.add_argument(
    "--moving_avg", type=int, default=25, help="window size of moving average"
)
parser.add_argument("--factor", type=int, default=3, help="attn factor")
parser.add_argument(
    "--distil",
    action="store_false",
    help="whether to use distilling in encoder, using this argument means not using distilling",
    default=True,
)
parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--output_attention",
    action="store_true",
    help="whether to output attention in ecoder",
)

# SparseTSF
parser.add_argument("--period_len", type=int, default=24, help="period length")
"""
Refer to the source repository, some default parameters are different for SparseTSF, as follows:
    --lradj type9 \
    --train_epochs 100 \
    --batch_size 128 \
    --patience 100 \
    --d_model 128 \
"""


# TimeXer
parser.add_argument(
    "--use_norm", type=int, default=1, help="whether to use normalize; True 1 False 0"
)
"""
Refer to the source repository, some default parameters are different for TimeXer, as follows:
    --factor 1 \
    --dropout 0.1 \
    --train_epochs 10 \
"""

# DeepEDM
parser.add_argument("--mlp_dropout", type=float, default=0.1)
parser.add_argument("--edm_dropout", type=float, default=0.1)
parser.add_argument("--activation_fn", type=str, default="selu")
parser.add_argument("--n_mlp_layers", type=int, default=1)
parser.add_argument("--n_edm_blocks", type=int, default=1, help="number of edm layers")
parser.add_argument(
    "--dist_projection_dim",
    type=int,
    default=64,
    help="dimension of projection for distance kernel",
)
parser.add_argument(
    "--add_pe",
    type=str_to_bool,
    default=True,
    help="whether to add positional encoding (true/false)",
)
parser.add_argument(
    "--theta", type=float, default=1.0, help="for edm which weighs different distances"
)
parser.add_argument("--delay", type=int, default=7, help="delay for edm")
parser.add_argument("--latent_channel_dim", type=int, default=-1)
parser.add_argument("--time_delay_stride", type=int, default=1)
parser.add_argument(
    "--n_proj_layers", type=int, default=1, help="number of projection layers"
)
parser.add_argument("--opt_type", type=str, default="Adam")
parser.add_argument("--min_lr", type=float, default=0.00001)
parser.add_argument("--reduce_lr_factor", type=float, default=0.8)
parser.add_argument("--weight_decay", type=float, default=0.00001)
parser.add_argument("--clip_grad_norm", type=float, default=1.0)
"""
Refer to the source repository, some default parameters are different for DeepEDM, as follows:
    --opt_type AdamW \
    --loss mae \
    --use_tdalign \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --patience 30 \
    --lradj type10 \
"""

# CrossLinear
parser.add_argument("--alpha", type=float, default=1, help="alpha")
parser.add_argument("--beta", type=float, default=1, help="beta")
"""
Refer to the source repository, some default parameters are different for CrossLinear, as follows:
"""

# Duet
parser.add_argument(
    "--hidden_size", type=int, default=256, help="Duet router hidden size"
)
parser.add_argument("--num_experts", type=int, default=4, help="Duet number of experts")
parser.add_argument(
    "--noisy_gating", type=int, default=1, help="Duet noisy gating; True 1 False 0"
)
parser.add_argument("--k", type=int, default=1, help="Duet top-k experts")
"""
Refer to the source repository, some default parameters are different for Duet, as follows:
    --factor 1 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --batch_size 256 \
    --lradj type8 \
    --learning_rate 0.02 \
    --train_epochs 100 \
    --num_workers 0 \
    --loss huber \
    --huber_delta 0.5 \
    --patience 10 \
"""

# ChannelTokenFormer
parser.add_argument(
    "--num_global_tokens",
    type=int,
    default=1,
    help="ChannelTokenFormer number of global tokens",
)
parser.add_argument(
    "--keep_prob",
    type=float,
    default=1,
    help="ChannelTokenFormer train time patch masking keep probability",
)
"""
Refer to the source repository, some default parameters are different for
ChannelTokenFormer, as follows:
    --factor 1 \
    --dropout 0.1 \
    --loss mse \
"""

# TSSI
parser.add_argument(
    "--n_res_layers", type=int, default=1, help="number of residual mlp layers"
)
parser.add_argument("--in_dim", type=int, default=128, help="input dimension of MLP")
parser.add_argument(
    "--hidden_dim", type=int, default=256, help="hidden dimension of MLP"
)
parser.add_argument(
    "--recent_factor",
    type=int,
    default=2,
    help="TSSI recent token length factor",
)
# optimization
parser.add_argument(
    "--num_workers", type=int, default=10, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=3, help="experiments run times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument(
    "--loss",
    type=str,
    choices=("mse", "mae", "huber"),
    default="mse",
    help="loss function",
)
parser.add_argument(
    "--huber_delta", type=float, default=1.0, help="delta for huber_loss"
)
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument("--pct_start", type=float, default=0.3, help="pct_start")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)
parser.add_argument(
    "--use_tdalign", action="store_true", help="use TDAlign loss", default=False
)

# GPU
parser.add_argument(
    "--use_gpu", type=str_to_bool, default=True, help="use gpu (true/false)"
)
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument(
    "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
)
parser.add_argument(
    "--devices", type=str, default="0,1", help="device ids of multile gpus"
)
# other setting
parser.add_argument(
    "--save_epoch_loss",
    type=str_to_bool,
    default=False,
    help="save epoch loss in training process (true/false)",
)

# multi-rate setting
parser.add_argument(
    "--res_list",
    nargs="+",
    type=int,
    default=[1],
    help="the list of resampling times, e.g., [1, 2, 3] means three times of resampling data",
)
parser.add_argument(
    "--ratio",
    nargs="+",
    type=int,
    default=[1],
    help="The data sampling ratio = highest / lower",
)
parser.add_argument(
    "--resample_data",
    type=str,
    default="./resample_data/ETTm1/1_4/1/",
    help="location of preprocess data with different sampling rate",
)


args = parser.parse_args()

if args.use_gpu:
    if args.use_multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices.replace(" ", "")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import torch

from exp.exp_main import Exp_Main
from models.MFCLR import MFCLR

torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


if not args.ratio:
    raise ValueError("invalid input args.ratio")

res_list = [int(res) for res in args.res_list]
res_itr_ratio = {
    res: {
        itr: {ratio: {"mse": 0.0, "mae": 0.0} for ratio in args.ratio}
        for itr in range(args.itr)
    }
    for res in res_list
}  # record mse and mae for each ratio in each run in each resampling

if args.is_training:
    print("Start training...")
    print("initial args in experiment:")
    print(args)
    print("")

    for t in res_list:
        res = str(t)
        args.resample_data_dir = os.path.join(args.resample_data, res)
        print(f"")
        print(
            f"=*=*=*=*=*=*=*=*=*= resampling time={res}, starting: =*=*=*=*=*=*=*=*=*="
        )
        # resample data to different sampling rates
        if not os.path.exists(args.resample_data_dir):
            raise ValueError(
                f"resample data directory {args.resample_data_dir} does not exist, please check the path."
            )
        else:
            with open(os.path.join(args.resample_data_dir, "vars_group.txt"), "r") as f:
                vars_group = ast.literal_eval(f.read())
            for ratio in args.ratio:
                print(f"ratio={ratio}, variates: {vars_group[ratio]}")
        print("")

        for ii in range(args.itr):
            itr = str(ii + 1)
            print(f"****************** itr={itr}, starting: ******************")
            if args.model == "MFCLR":
                mses, maes = MFCLR(args, res, itr)
            else:
                exp = Exp_Main(args)
                exp.train(res, itr)
                mses, maes = exp.test(
                    res, itr
                )  # the value list of mse and mae emcompassing all ratios
                del exp
            for ratio in args.ratio:
                ratio_mse = mses[args.ratio.index(ratio)]
                ratio_mae = maes[args.ratio.index(ratio)]
                res_itr_ratio[t][ii][ratio]["mse"] = ratio_mse
                res_itr_ratio[t][ii][ratio]["mae"] = ratio_mae
                # print(f"{ratio}: mse={ratio_mse:.3f}, mae={ratio_mae:.3f}")
            torch.cuda.empty_cache()
            gc.collect()
            print(f"****************** itr={itr}, ending ******************")
            print("")
        print(f"=*=*=*=*=*=*=*=*=*= resampling time={res}, ending: =*=*=*=*=*=*=*=*=*=")
        print("")
else:
    print("Start predicting...")
    print("initial args in experiment:")
    print(args)
    print("")

    for t in res_list:
        res = str(t)
        args.resample_data_dir = os.path.join(args.resample_data, res)
        print(f"")
        print(
            f"=*=*=*=*=*=*=*=*=*= resampling time={res}, starting: =*=*=*=*=*=*=*=*=*="
        )
        # resample data to different sampling rates
        if not os.path.exists(args.resample_data_dir):
            raise ValueError(
                f"resample data directory {args.resample_data_dir} does not exist, please check the path."
            )
        else:
            with open(os.path.join(args.resample_data_dir, "vars_group.txt"), "r") as f:
                vars_group = ast.literal_eval(f.read())
            for ratio in args.ratio:
                print(f"ratio={ratio}, variates: {vars_group[ratio]}")
        print("")

        for ii in range(args.itr):
            itr = str(ii + 1)
            print(f"****************** itr={itr}, starting: ******************")
            if args.model == "MFCLR":
                print("MFCLR not support testing only currently.")
            else:
                exp = Exp_Main(args)
                mses, maes = exp.predict(
                    res, itr
                )  # the value list of mse and mae emcompassing all ratios
                del exp
                for ratio in args.ratio:
                    ratio_mse = mses[args.ratio.index(ratio)]
                    ratio_mae = maes[args.ratio.index(ratio)]
                    res_itr_ratio[t][ii][ratio]["mse"] = ratio_mse
                    res_itr_ratio[t][ii][ratio]["mae"] = ratio_mae

            torch.cuda.empty_cache()
            gc.collect()
            print(f"****************** itr={itr}, ending ******************")
            print("")
        print(f"=*=*=*=*=*=*=*=*=*= resampling time={res}, ending: =*=*=*=*=*=*=*=*=*=")
        print("")

print(
    f"All experiments end!\nresample_times={len(res_list)}, itr_num={args.itr}, resample_ratios={args.ratio}.\nList all results:"
)
for t in res_list:
    print(f"resampling time={t}:")
    for ii in range(args.itr):
        print(f"\titr={ii+1}:")
        for ratio in args.ratio:
            print(
                f"\t\tratio={ratio}: mse={res_itr_ratio[t][ii][ratio]['mse']:.3f}, mae={res_itr_ratio[t][ii][ratio]['mae']:.3f}"
            )
print("")

print(f">>>>>>>>>>>> Statistical results: >>>>>>>>>>>>")
print("")
# print("for each time of resampling data:")
all_res_mse = []
all_res_mae = []
for t in res_list:
    res_avg_mse = []  # form itr_num * ratios_num experiments
    res_avg_mae = []
    print(f"current resampling time={t}:")
    for ii in range(args.itr):
        for ratio in args.ratio:
            res_avg_mse.append(res_itr_ratio[t][ii][ratio]["mse"])
            res_avg_mae.append(res_itr_ratio[t][ii][ratio]["mae"])
    print(
        f"\t\taverage_mse={np.mean(res_avg_mse):.3f}, average_mae={np.mean(res_avg_mae):.3f}"
    )
    all_res_mse.append(np.mean(res_avg_mse))
    all_res_mae.append(np.mean(res_avg_mae))
print(f"\tmin_mse={np.min(all_res_mse):.3f}, min_mae={np.min(all_res_mae):.3f}")
print(f"\tmax_mse={np.max(all_res_mse):.3f}, max_mae={np.max(all_res_mae):.3f}")
print(
    f"\taverage_mse={np.mean(all_res_mse):.3f}±{np.std(all_res_mse):.3f}, average_mae={np.mean(all_res_mae):.3f}±{np.std(all_res_mae):.3f}"
)
print("")
