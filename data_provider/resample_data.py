import argparse
import os
import random

import pandas as pd



parser = argparse.ArgumentParser(description="resample data to different sampling rates")
parser.add_argument(
    "--root_path",
    type=str,
    default="./datasets/ETT-small",
    help="root path of the data file",
)
parser.add_argument("--data_path", type=str, default="ETTm1.csv", help="data file")
parser.add_argument(
    "--resample_type",
    type=str,
    default="iloc",
    help="the method of resampling different datasets, options: [iloc, sum]",
)
parser.add_argument(
    "--ratio",
    type=int,
    nargs="+",
    default=[1, 4],
    help="The data sampling ratio = highest / lower",
)
parser.add_argument(
    "--resample_data",
    type=str,
    default="./resample_data/ETTm1/1_4/",
    help="location of preprocess data with different sampling rate",
)
parser.add_argument(
    "--resample_times",
    type=int,
    default=3,
    help="number of times to resample the data",
)
parser.add_argument(
    "--fix_seed",
    type=int,
    default=2025,
    help="random seed for shuffling variables",
)

args = parser.parse_args()
data_path = os.path.join(args.root_path, args.data_path)
if not os.path.exists(data_path):
    raise ValueError(f"data path {data_path} does not exist, please check the path.")


max_ratio = max(args.ratio)  # int: the lowest rate
n_ratio = len(args.ratio)

# Split train, validation, and test data before resampling.
initial_data = pd.read_csv(data_path)
least_obs = int(len(initial_data) / max_ratio)  # the least number of observations
vars_num = len(initial_data.columns) - 1  # number of variates
initial_data = initial_data.iloc[: max_ratio * least_obs]
least_train_obs = int(least_obs * 0.7)
least_test_obs = int(least_obs * 0.2)
train_data = initial_data.iloc[: least_train_obs * max_ratio]
test_data = initial_data.iloc[-least_test_obs * max_ratio :]
vali_data = initial_data.iloc[least_train_obs * max_ratio : -least_test_obs * max_ratio]
data_splits = {"train": train_data, "vali": vali_data, "test": test_data}

group_sizes = [vars_num // n_ratio] * n_ratio
group_sizes[-1] += vars_num % n_ratio

print(f"all variates initial train timestamps: {len(train_data)}")
print(f"all variates initial vali timestamps: {len(vali_data)}")
print(f"all variates initial test timestamps: {len(test_data)}")
print(f"variable group sizes: {group_sizes}")
print("")

vars_list = list(range(1, vars_num + 1))
vars_group = {"all": vars_list}

for t in range(args.resample_times):
    print("========================================")
    print(f"Resampling time {t+1} start:")
    random.seed(args.fix_seed + t)
    random.shuffle(vars_list)
    resample_time_dir = os.path.join(args.resample_data, str(t + 1))
    for i, ratio in enumerate(args.ratio):
        start = i * (vars_num // n_ratio)
        end = start + group_sizes[i]
        vars_group[ratio] = vars_list[start:end]
        print(f"ratio={ratio}, variates: {vars_group[ratio]}")

        each_ratio_dir = os.path.join(resample_time_dir, str(ratio))
        os.makedirs(each_ratio_dir, exist_ok=True)

        for flag, data in data_splits.items():
            col_selected = data.iloc[:, [0] + vars_group[ratio]]

            if args.resample_type == "sum":
                new_data = col_selected.groupby(col_selected.index // ratio).agg(
                    {
                        "date": "last",
                        **{col: "sum" for col in col_selected.columns if col != "date"},
                    }
                )
            elif args.resample_type == "iloc":
                new_data = col_selected.iloc[::ratio]
            else:
                raise ValueError(f"Unsupported resample_type: {args.resample_type}")

            new_data.to_csv(
                os.path.join(each_ratio_dir, f"{flag}.csv"),
                index=False,
            )
            print(f"\t{flag} timestamps: {len(new_data)}")

    print(
        f"Resampling time {t+1} end: train, vali, test data saved in {resample_time_dir}."
    )
    with open(os.path.join(resample_time_dir, "vars_group.txt"), "w") as f:
        f.write(str(vars_group))
