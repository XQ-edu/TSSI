import argparse
import os
import random
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from layers.encoder import BackboneEncoder, Projector
from utils.data_augmentation import data_generation_multi
from utils.lossfunc import combine_loss
from utils.metrics import instance_averaged_metrics
from utils.util_func import (
    centerize_vary_length_series,
    split_with_nan,
    take_per_row_multigrain,
    torch_pad_nan,
)

warnings.filterwarnings("ignore")
abs_path = os.path.dirname(os.path.realpath(__file__))

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.set_num_threads(6)


class MF_CLR:
    def __init__(
        self,
        input_dims,
        grain_split,
        total_dim,
        output_dims=None,
        hidden_dims=256,
        depth=7,
        device="cpu",
        lr=0.001,
        batch_size=64,
        max_train_length=None,
        temporal_unit=0,
        ph_dim=128,
        after_iter_callback=None,
        after_epoch_callback=None,
        projection=True,
        da="proposed",
    ):
        """Initialize a MF_CLR model.
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (list): The representation dimension on each granularity.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        """
        super().__init__()
        self.device = device
        if output_dims is not None:
            self.output_dim_list = [output_dims for i in range(len(grain_split) - 1)]
        else:
            self.output_dim_list = [320 for i in range(len(grain_split) - 1)]
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.grain_split_list = grain_split

        assert len(self.output_dim_list) == len(self.grain_split_list) - 1
        self._net_list, self.net_list = [], []
        for l in range(len(grain_split) - 1):
            if len(self.net_list) == 0:
                input_dim_l = input_dims
                output_dim_l = self.output_dim_list[0]
            else:
                if projection is False:
                    input_dim_l = self.output_dim_list[l - 1]
                    output_dim_l = self.output_dim_list[l]
                else:
                    input_dim_l = self.output_dim_list[l - 1] + ph_dim * l
                    output_dim_l = self.output_dim_list[l] + ph_dim * l
            _net = BackboneEncoder(
                input_dims=input_dim_l,
                output_dims=output_dim_l,
                hidden_dims=hidden_dims,
                depth=depth,
            ).to(self.device)
            net = torch.optim.swa_utils.AveragedModel(_net)
            net.update_parameters(_net)
            self._net_list.append(_net)
            self.net_list.append(net)

        self.ph_list = []
        for l in range(len(grain_split) - 1):
            projector = Projector(
                input_dims=grain_split[l + 1] - grain_split[l],
                h1=int(ph_dim / 2),
                h2=ph_dim,
            ).to(self.device)
            self.ph_list.append(projector)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.projection = projection

        assert da in [
            "proposed",
            "scaling",
            "shifting",
            "jittering",
            "permutation",
            "random mask",
        ]
        self.da_method = da

        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        """Training the MF-CLR model.

        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.

        Returns:
            loss_log: a list containing the training losses on each epoch.
        """
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = (
                200 if train_data.size <= 100000 else 600
            )  # default param for n_iters

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(
                    split_with_nan(train_data, sections, axis=1), axis=0
                )

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)

        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=False,
            drop_last=True,
        )
        optimizer_list, scheduler_list = [], []
        for g in range(len(self.grain_split_list) - 1):
            optimizer = torch.optim.AdamW(
                list(self._net_list[g].parameters())
                + list(self.ph_list[g].parameters()),
                lr=self.lr,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.7
            )
            optimizer_list.append(optimizer)
            scheduler_list.append(scheduler)

        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if (
                    self.max_train_length is not None
                    and x.size(1) > self.max_train_length
                ):
                    window_offset = np.random.randint(
                        x.size(1) - self.max_train_length + 1
                    )
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)
                grain_data_list = take_per_row_multigrain(
                    x,
                    np.random.randint(low=0, high=1, size=x.size(0)),
                    ts_l,
                    self.grain_split_list,
                )

                for g in range(len(self.grain_split_list) - 1):

                    if g == 0:
                        fine_grained = grain_data_list[0]
                        coarse_grained = grain_data_list[1]
                    else:
                        fine_grained = out_embed
                        coarse_grained = grain_data_list[g + 1]

                    this_optimiser = optimizer_list[g]
                    this_optimiser.zero_grad()

                    fine_grained = fine_grained.to(self.device)

                    out = self._net_list[g](fine_grained)
                    time_start = time.time()
                    fine_grained_aug = torch.from_numpy(
                        data_generation_multi(fine_grained.cpu().numpy())
                    ).float()
                    fine_grained_aug = fine_grained_aug.to(self.device)
                    out_aug = self._net_list[g](fine_grained_aug)
                    coarse_grained = coarse_grained.to(self.device)

                    grain_loss = combine_loss(
                        out, out_aug, coarse_grained, coarse_grained
                    )
                    grain_loss.backward()

                    cum_loss += grain_loss.item()

                    fine_embed = out.clone().detach().cpu().numpy()
                    if self.projection is True:
                        coarse_embed = self.ph_list[g](coarse_grained)
                        coarse_embed = coarse_embed.clone().detach().cpu().numpy()
                        out_embed = np.concatenate([fine_embed, coarse_embed], axis=2)
                    else:
                        out_embed = np.concatenate([fine_embed, coarse_embed], axis=2)
                    out_embed = torch.from_numpy(out_embed)

                    this_optimiser.step()
                    self.net_list[g].update_parameters(self._net_list[g])

                    n_epoch_iters += 1
                    self.n_iters += 1

                    if self.after_iter_callback is not None:
                        self.after_iter_callback(self, grain_loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            for each_scheduler in scheduler_list:
                each_scheduler.step()

        return loss_log

    def _eval_with_pooling(
        self, x, encoding_net, mask=None, slicing=None, encoding_window=None
    ):

        out = encoding_net(x.to(self.device, non_blocking=True), mask)

        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(
        self,
        data,
        mask=False,
        encoding_window=None,
        casual=False,
        sliding_length=None,
        sliding_padding=0,
        batch_size=None,
    ):
        """Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        """

        for each_net in self.net_list:
            assert each_net is not None, "please train or load a net first"
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training_list = []
        for each_net in self.net_list:
            org_training_list.append(each_net.training)
            each_net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1,
                        )
                        for g in range(len(self.grain_split_list) - 1):
                            if g == 0:
                                encoding_x_sliding = x_sliding[
                                    :, :, : self.grain_split_list[0]
                                ]
                                encoding_f_sliding = x_sliding[
                                    :,
                                    :,
                                    self.grain_split_list[0] : self.grain_split_list[1],
                                ]
                                encoding_net = self.net_list[0]
                                out_g = self._eval_with_pooling(
                                    encoding_x_sliding,
                                    encoding_net,
                                    slicing=slice(
                                        sliding_padding,
                                        sliding_padding + sliding_length,
                                    ),
                                    encoding_window=encoding_window,
                                )
                                ph_net = self.ph_list[0]
                                out_f = ph_net(encoding_f_sliding)
                            else:
                                encoding_x_sliding = out_g
                                encoding_f_sliding = out_f
                                encoding_x_sliding = torch.from_numpy(
                                    np.concatenate(
                                        [
                                            encoding_x_sliding.numpy(),
                                            encoding_f_sliding.numpy(),
                                        ],
                                        axis=2,
                                    )
                                )
                                encoding_net = self.net_list[g]
                                out_g = self._eval_with_pooling(
                                    encoding_x_sliding,
                                    encoding_net,
                                    slicing=slice(
                                        sliding_padding,
                                        sliding_padding + sliding_length,
                                    ),
                                    encoding_window=encoding_window,
                                )
                        reprs.append(out_g)

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == "full_series":
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)

                else:
                    for g in range(len(self.grain_split_list) - 1):

                        if g == 0:
                            encoding_x = x[:, :, : self.grain_split_list[0]]
                            encoding_net = self.net_list[0]
                            out_g = self._eval_with_pooling(
                                encoding_x,
                                encoding_net,
                                mask,
                                encoding_window=encoding_window,
                            )

                            ph_net = self.ph_list[0].to(self.device)
                            encoding_f = x[
                                :,
                                :,
                                self.grain_split_list[0] : self.grain_split_list[1],
                            ].to(self.device)
                            out_f = ph_net(encoding_f)

                        else:
                            encoding_x = out_g.clone().detach().numpy()
                            ph_f = out_f.clone().detach().cpu().numpy()
                            encoding_x = np.concatenate([encoding_x, ph_f], axis=2)
                            encoding_x = torch.from_numpy(encoding_x)
                            encoding_net = self.net_list[g]
                            out_g = self._eval_with_pooling(
                                encoding_x,
                                encoding_net,
                                mask,
                                encoding_window=encoding_window,
                            )

                            ph_net = self.ph_list[g].to(self.device)
                            encoding_f = x[
                                :,
                                :,
                                self.grain_split_list[g] : self.grain_split_list[g + 1],
                            ].to(self.device)
                            out_f = ph_net(encoding_f)
                    out = np.concatenate([out_g.cpu(), out_f.cpu()], axis=2)
                    out = torch.from_numpy(out)
                    if encoding_window == "full_series":
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        for i in range(len(self.net_list)):
            self.net_list[i].train(org_training_list[i])

        return output.numpy()

    def save(self, file_name):
        """Save the model to a file.
        Args:
            fn (str): filename.
        """
        param_dict = {}
        for g in range(len(self.grain_split_list) - 1):
            param_dict["encoder_" + str(g)] = self.net_list[g].state_dict()
            param_dict["projector_" + str(g)] = self.ph_list[g].state_dict()
        torch.save(param_dict, file_name)

    def load(self, file_name):
        """Load the model from a file.
        Args:
            fn (str): filename.
        """
        param_dict = torch.load(file_name)
        for g in range(len(self.grain_split_list) - 1):
            self.net_list[g].load_state_dict(param_dict["encoder_" + str(g)])
            self.ph_list[g].load_state_dict(param_dict["projector_" + str(g)])


class Public_Dataset_MFCLR:
    def __init__(self, args, res, itr):
        """
        for multivariate time series representation learning and forecasting, some parameters in the source code are not used or replaced, such as:
        self.origin_file = args.dataset
        self.test_file = args.dataset + "_processed_OT=" + args.ot_granu
        self.data = args.ds
        self.fcst_granu = args.ot_granu
        self.saving = args.save
        self.y_dim
        self.pred_len_ori
        self.seq_len = args.lookback = args.seq_len
        """
        self.enc_len = (
            args.enc_len
        )  # for spliting the data into non-overlapping samples for representation learning
        self.seq_len = (
            args.seq_len
        )  # for forecasting setting, sequence length. this is the longest sequence length crossponding to the highest sampling rate, for lower sampling rate, the actual sequence length is equal to args.seq_len // ratio
        self.pred_len = (
            args.pred_len
        )  # for forecasting setting, prediction length. this is the longest sequence length crossponding to the highest sampling rate, for lower sampling rate, the actual sequence length is equal to args.pred_len // ratio
        self.epoch = args.train_epochs
        self.training = True if args.is_training == 1 else False
        self.learning_rate = args.learning_rate
        self.projector_dim = args.ph_dim
        self.batch_size = args.batch_size
        self.output_dims = args.out_dim
        self.hidden_dims = args.hidden_dim
        self.depth = args.depth

        # add some parameters for multivariate time series representation learning and forecasting
        self.total_dim = args.enc_in
        """
        ratio: a list of int numbers, the data sampling ratio = highest / lower
        and for representation learning, it needs to be sorted from highest to lowest (e.g., the value from small to large, such as [1, 4])
        """
        self.ratio = sorted(args.ratio, reverse=False)
        self.checkpoints = args.checkpoints
        self.results = args.results
        self.res = res
        self.itr = itr

        """
        grain_split_list: cut dimension from **highest to lowest** sampling rate, for example, [3,7], This means that the first three variables are sampled consistently and at the highest rate, and the last four variables are sampled consistently and at the lowest rate.
        """
        if args.ratio[1] > args.ratio[0]:
            d = args.enc_in // len(args.ratio)
            self.grain_split_list = [d * (i + 1) for i in range(len(args.ratio) - 1)]
            self.grain_split_list.append(args.enc_in)
        else:
            d = args.enc_in // len(args.ratio)
            self.grain_split_list = [
                args.enc_in - d * (len(args.ratio) - 1 - i)
                for i in range(len(args.ratio) - 1)
            ]
            self.grain_split_list.append(args.enc_in)

        if args.use_gpu is False:
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.resample_data_dir = args.resample_data_dir

    def gen_enc_array(self, flag, keep_date=False):
        data_array = []
        for i, ratio in enumerate(self.ratio):
            data_path = os.path.join(self.resample_data_dir, f"{ratio}", f"{flag}.csv")
            df_raw = pd.read_csv(data_path)
            if keep_date is False:
                df_raw.drop(columns=["date"], inplace=True)
            df_data = df_raw.values
            # need repeat data to align the seq_len and pred_len of all variables to the variable with the highest sampling rate
            data_array.append(np.repeat(df_data, ratio, axis=0))
        data_array = np.concatenate(data_array, axis=1)
        return data_array

    def data_split(self, data_array_norm, _method="cut"):
        # this function is used to split the data into non-overlapping samples for representation learning, only for train and vali data
        assert _method in ["slide", "cut"]
        if _method == "cut":
            cut_rows = (
                int(np.floor(data_array_norm.shape[0] / self.enc_len)) * self.enc_len
            )
            data_array_norm = data_array_norm[-cut_rows:, :]
            win_num = data_array_norm.shape[0] // self.enc_len
            reconstruct_data = []
            for i in range(win_num):
                start = data_array_norm.shape[0] - (win_num - i) * self.enc_len
                end = start + self.enc_len
                reconstruct_data.append(data_array_norm[start:end, :])
            return np.array(reconstruct_data)
        elif _method == "slide":
            win_num = data_array_norm.shape[0] - self.enc_len + 1
            reconstruct_data = []
            for i in range(win_num):
                start = i
                end = start + self.enc_len
                reconstruct_data.append(data_array_norm[start:end, :])
            return np.array(reconstruct_data)

    def normalisation(self, inputArray, meanlist=None, stdlist=None):
        if meanlist is None and stdlist is None:
            meanlist, stdlist = [], []
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append(
                    (inputArray[:, j] - np.mean(inputArray[:, j]))
                    / np.std(inputArray[:, j])
                )
                meanlist.append(np.mean(inputArray[:, j]))
                stdlist.append(np.std(inputArray[:, j]))
            return np.array(normalised_array).T, meanlist, stdlist
        else:
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - meanlist[j]) / stdlist[j])
            return np.array(normalised_array).T, meanlist, stdlist

    def encoding(self, data_array_encode, flag=None):
        print("fitting input size : ", data_array_encode.shape)
        model = MF_CLR(
            input_dims=self.grain_split_list[0],
            grain_split=self.grain_split_list,
            total_dim=self.total_dim,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            ph_dim=self.projector_dim,
            device=self.device,
            batch_size=self.batch_size,
            lr=self.learning_rate,
        )
        ratio_str = "_".join([str(ratio) for ratio in self.ratio])
        model_path = os.path.join(self.checkpoints, self.res, self.itr, ratio_str)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if flag != "test":
            loss_log = model.fit(
                train_data=data_array_encode,
                n_epochs=self.epoch,
                verbose=True,
            )
            model.save(model_path + "/" + "repr_learning.pkl")
        else:
            model.load(model_path + "/" + "repr_learning.pkl")

        print("encoding input size : ", data_array_encode.shape)
        padding = 0
        repr = model.encode(
            data_array_encode,
            casual=False,
            sliding_length=None,
            sliding_padding=padding,
        )
        print(
            f"origin {flag}_repr.shape : ", repr.shape
        )  # [instance_num, length, repr dim for all variates]
        return repr

    def test_instance_construct(self, mean, std, keep_date=False):
        """
        construct the test instances:
        seq_x is an aligned array: the length of each instance is equal to the longest sequence length
        seq_y is a list: each element is an array, the length of each element is equal to the pred_len of each ratio
        instance_idx is a list: the unique index of each instance for each ratio
        """
        x, y = [], []
        for i, r in enumerate(self.ratio):
            data_path = os.path.join(self.resample_data_dir, f"{r}/test.csv")
            df_raw = pd.read_csv(data_path)
            if keep_date is False:
                df_raw.drop(columns=["date"], inplace=True)
            df_data = df_raw.values
            var_num = df_data.shape[1]
            if i == 0:
                for j in range(var_num):
                    df_data[:, j] = (df_data[:, j] - mean[j]) / std[j]
            else:
                for j in range(var_num):
                    df_data[:, j] = (
                        df_data[:, j]
                        - mean[self.grain_split_list[i - 1] : self.grain_split_list[i]][
                            j
                        ]
                    ) / std[self.grain_split_list[i - 1] : self.grain_split_list[i]][j]

            # sliding window
            seq_len = self.seq_len // r
            pred_len = self.pred_len // r
            win_num = df_data.shape[0] - seq_len - pred_len + 1
            x.append(
                np.stack([df_data[i : i + seq_len] for i in range(win_num)], axis=0)
            )
            y.append(
                np.stack(
                    [
                        df_data[i + seq_len : i + seq_len + pred_len]
                        for i in range(win_num)
                    ],
                    axis=0,
                )
            )
            # the input x needs to be aligned length with the largest sequence length
            x[i] = np.repeat(x[i], r, axis=1)
        # second, get instances for each ratio and label the instance with unique index
        seq_x, seq_y, instance_idx = [], [], []
        instance_num = max([len(x[i]) for i in range(len(self.ratio))])
        for i, r in enumerate(self.ratio):
            curr_x, curr_y, curr_idx = [], [], []
            if i == 0:
                curr_x = x[i]
                curr_y = y[i]
                curr_idx = np.arange(instance_num)
            else:
                for j in range(instance_num):
                    idx = (j - 1) // r + 1
                    curr_x.append(x[i][idx])
                    curr_y.append(y[i][idx])
                    curr_idx.append(idx)
                curr_x = np.stack(curr_x, axis=0)
                curr_y = np.stack(curr_y, axis=0)
            seq_x.append(curr_x)
            seq_y.append(curr_y)  # each element is a list
            instance_idx.append(curr_idx)

        seq_x = np.concatenate(seq_x, axis=-1)
        return seq_x, seq_y, instance_idx

    def dataset_construct(self, inputArray, repr, input_app=True, _shuffle=True):
        # construct train and vali instances for fine-tuning
        min_length = min(inputArray.shape[0], repr.shape[0])
        inputArray = inputArray[-min_length:]
        repr = repr[-min_length:]
        print(f"truncated size: {inputArray.shape},  {repr.shape}")

        # Combine input and representation if needed
        data = np.concatenate((inputArray, repr), axis=1) if input_app else inputArray

        # sliding window
        win_num = data.shape[0] - self.seq_len - self.pred_len + 1
        seq_x = np.stack([data[i : i + self.seq_len] for i in range(win_num)], axis=0)
        seq_y = np.stack(
            [
                inputArray[i + self.seq_len : i + self.seq_len + self.pred_len]
                for i in range(win_num)
            ],
            axis=0,
        )

        if _shuffle is True:
            random_idx = np.random.permutation(seq_x.shape[0])
            seq_x = seq_x[random_idx]
            seq_y = seq_y[random_idx]
        return seq_x, seq_y

    def fit_ridge(
        self, train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000
    ):
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_features.shape[0] > MAX_SAMPLES:
            split = train_test_split(
                train_features, train_y, train_size=MAX_SAMPLES, random_state=0
            )
            train_features = split[0]
            train_y = split[2]
        if valid_features.shape[0] > MAX_SAMPLES:
            split = train_test_split(
                valid_features, valid_y, train_size=MAX_SAMPLES, random_state=0
            )
            valid_features = split[0]
            valid_y = split[2]

        alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        valid_results = []
        for alpha in alphas:
            lr = Ridge(alpha=alpha).fit(train_features, train_y)
            valid_pred = lr.predict(valid_features)
            score = (
                np.sqrt(((valid_pred - valid_y) ** 2).mean())
                + np.abs(valid_pred - valid_y).mean()
            )
            valid_results.append(score)
            print(f"alpha={alpha}, score={score}")
        best_alpha = alphas[np.argmin(valid_results)]

        lr = Ridge(alpha=best_alpha)
        lr.fit(train_features, train_y)
        print(f"best alpha={best_alpha}")
        return lr

    def cal_loss(self, preds, labels):
        assert np.ndim(preds) == np.ndim(labels) == 3
        assert preds.shape[0] == labels.shape[0]
        mse = np.mean((preds - labels) ** 2)
        mae = np.mean(np.abs(preds - labels))
        mape = np.mean(np.abs(preds - labels) / labels)
        return mse, mae, mape

    def fcst(self):
        # encoding
        data_train_array = self.gen_enc_array(flag="train")
        data_vali_array = self.gen_enc_array(flag="vali")

        data_train_array_norm, mean, std = self.normalisation(data_train_array)
        data_vali_array_norm, _, _ = self.normalisation(data_vali_array, mean, std)
        del data_train_array, data_vali_array
        print("data_train_array_norm shape : ", data_train_array_norm.shape)
        print("data_vali_array_norm shape : ", data_vali_array_norm.shape)

        """
        step1: split non-overlapping instances (length of each instance is enc_len -- round down) from train and vali data, for representation learning
        """
        data_train_array_enc = self.data_split(data_train_array_norm)
        data_vali_array_enc = self.data_split(data_vali_array_norm)
        print(
            "after splitting, data_train_array_enc shape : ", data_train_array_enc.shape
        )
        print(
            "after splitting, data_vali_array_enc shape : ", data_vali_array_enc.shape
        )
        print()
        """
        step2: use non-overlapping train_vali instances to train the representation learning model, and get the representation of each time step of train and vali data. [instance_num, length, variate_num] --> [instance_num, length, repr dim for all variates]
        """
        data_repr_learn = np.concatenate(
            (data_train_array_enc, data_vali_array_enc), axis=0
        )
        train_vali_repr = self.encoding(data_repr_learn, flag="train_vali")
        # get representation of train and valid data, each time step has its representation embeded by all variables
        train_repr = train_vali_repr[: data_train_array_enc.shape[0]].reshape(
            -1, train_vali_repr.shape[-1]
        )
        vali_repr = train_vali_repr[data_train_array_enc.shape[0] :].reshape(
            -1, train_vali_repr.shape[-1]
        )
        print(f"reshape train_repr shape : ", train_repr.shape)
        print(f"reshape vali_repr shape : ", vali_repr.shape)
        print()

        """
        step3: construct the train and vali data for fine-tuning: sliding window on train and vali data with representation of each time step
        """
        train_x, train_y = self.dataset_construct(data_train_array_norm, train_repr)
        del data_train_array_enc, data_train_array_norm, train_repr
        print("all ratio, train, instances: ", train_x.shape[0])
        print("train_x shape : ", train_x.shape)
        print("train_y shape : ", train_y.shape)
        vali_x, vali_y = self.dataset_construct(data_vali_array_norm, vali_repr)
        del data_vali_array_enc, data_vali_array_norm, vali_repr
        print("all ratio, vali, instances: ", vali_x.shape[0])
        print("vali_x shape : ", vali_x.shape)
        print("vali_y shape : ", vali_y.shape)
        print()
        """
        step4: fine-tuning the forecasting model
        """
        r_train_x, r_train_y = train_x.reshape(train_x.shape[0], -1), train_y.reshape(
            train_y.shape[0], -1
        )
        print("reshape train_x shape : ", r_train_x.shape)
        print("reshape train_y shape : ", r_train_y.shape)
        r_vali_x, r_vali_y = vali_x.reshape(vali_x.shape[0], -1), vali_y.reshape(
            vali_y.shape[0], -1
        )
        print("reshape vali_x shape : ", r_vali_x.shape)
        print("reshape vali_y shape : ", r_vali_y.shape)
        print()

        t = time.time()
        predict_model = self.fit_ridge(r_train_x, r_train_y, r_vali_x, r_vali_y)
        fit_time = time.time() - t
        print("fit time : ", fit_time)
        del train_x, train_y, r_train_x, r_train_y
        del vali_x, vali_y, r_vali_x, r_vali_y

        """
        step5: construct the test instances: normalising and sliding window on test data
        """
        test_x_array, test_y_list_array, test_instance_idxs = (
            self.test_instance_construct(mean, std)
        )
        print("the normalized and aligned test_x_array shape : ", test_x_array.shape)
        """
        step6: get the representation of the test instances's test_x_array
        """

        test_x_repr = self.encoding(test_x_array, flag="test")
        """
        step7: combine the test_x_array with its representation
        """
        test_x = np.concatenate((test_x_array, test_x_repr), axis=-1)
        print("all ratios, test, instances: ", test_x.shape[0])
        print(
            "combine test_x_array with its representation test_x_repr, test_x shape : ",
            test_x.shape,
        )
        print()

        """
        step8: forecasting
        """
        r_test_x = test_x.reshape(test_x.shape[0], -1)
        print("reshape test_x shape : ", r_test_x.shape)
        print()
        t = time.time()

        r_test_pred = predict_model.predict(r_test_x)
        forecast_time = time.time() - t
        print("forecasting time : ", forecast_time)
        test_pred = r_test_pred.reshape(test_x.shape[0], self.pred_len, -1)
        print("aligned test_pred shape : ", test_pred.shape)
        print()

        mses = []
        maes = []
        for i, r in enumerate(self.ratio):
            s_begin = 0 if i == 0 else self.grain_split_list[i - 1]
            s_end = self.grain_split_list[i]

            preds = test_pred[:, :, s_begin:s_end]
            B, L, C = preds.shape
            preds = preds.reshape(B, L // r, r, C).mean(axis=2)
            trues = test_y_list_array[i]
            inputxs = test_x_array[:, ::r, s_begin:s_end]
            instance_idxs = test_instance_idxs[i]

            # result save
            results_path = os.path.join(self.results, self.res, self.itr, str(r))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            print(f"ratio:{r}, test, unique instances:{len(np.unique(instance_idxs))}")
            mae, mse, rmse, mape, mspe, rse, corr = instance_averaged_metrics(
                preds, trues, instance_idxs
            )
            print("ratio:{}, mse:{}, mae:{}".format(r, mse, mae))
            mses.append(mse)
            maes.append(mae)
            np.save(
                results_path + "/" + "metrics.npy",
                np.array([mae, mse, rmse, mape, mspe, rse]),
            )
            np.save(results_path + "/" + "preds.npy", preds)
            np.save(results_path + "/" + "trues.npy", trues)
            np.save(results_path + "/" + "inputxs.npy", inputxs)
            np.save(results_path + "/" + "instance_idxs.npy", instance_idxs)
            del preds, trues, inputxs, instance_idxs
        return mses, maes


def MFCLR(args, res, itr):
    # the following parameters are essential for MFCLR, and are set according to the source code.
    args.ph_dim = 32  # dimension of projector
    args.out_dim = 64  # dimension of output
    args.hidden_dim = 128  # dimension of hidden
    args.depth = 4  # numbers of dilated convolutional layers
    args.enc_len = 128  # for spliting the data into non-overlapping samples for representation learning

    args.use_gpu = True if torch.cuda.is_available() else False

    CLR_EXP = Public_Dataset_MFCLR(args, res, itr)
    mses, maes = CLR_EXP.fcst()
    return mses, maes
