import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import (
    TSSI,
    ChannelTokenFormer,
    CrossLinear,
    DeepEDM,
    DLinear,
    Duet,
    SparseTSF,
    TimeXer,
    iTransformer,
)
from utils.data_alignment import data_alignment
from utils.metrics import instance_averaged_metrics
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):  # build model
        model_dict = {
            "DLinear": DLinear,
            "iTransformer": iTransformer,
            "SparseTSF": SparseTSF,
            "TSSI": TSSI,
            "TimeXer": TimeXer,
            "DeepEDM": DeepEDM,
            "CrossLinear": CrossLinear,
            "ChannelTokenFormer": ChannelTokenFormer,
            "Duet": Duet,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        num_params_model = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters in model: {num_params_model}")
        return model

    def _get_data(self, flag, ratio):
        # ratio is a list of int numbers
        data_set, data_loader = data_provider(self.args, flag, ratio)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.opt_type == "Adam":
            model_optim = optim.Adam(
                [*self.model.parameters()], lr=self.args.learning_rate
            )
        elif self.args.opt_type == "AdamW":
            model_optim = optim.AdamW(
                [*self.model.parameters()],
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.opt_type}")
        return model_optim

    def _compute_loss(self, pred, true):
        # pred and true are lists of tensors shaped as
        # [batch_size, self.args.pred_len // ratio, corresponding enc_in].
        loss = {
            "loss_opt": torch.tensor(0.0).to(self.device),
            **{
                ratio: {
                    "loss_mse": torch.tensor(0.0).to(self.device),
                    "loss_mae": torch.tensor(0.0).to(self.device),
                    "loss_huber": torch.tensor(0.0).to(self.device),
                }
                for ratio in self.args.ratio
            },
        }

        loss_opt = torch.tensor(0.0).to(self.device)
        total_elements = 0

        for i, ratio in enumerate(self.args.ratio):
            loss_mse = torch.mean(
                (pred[i] - true[i]) ** 2, dim=1
            )  # [batch_size, corresponding enc_in]
            loss_mae = torch.mean(torch.abs(pred[i] - true[i]), dim=1)
            loss_huber = torch.mean(
                F.huber_loss(
                    pred[i], true[i], reduction="none", delta=self.args.huber_delta
                ),
                dim=1,
            )

            loss[ratio]["loss_mse"] = torch.mean(loss_mse)
            loss[ratio]["loss_mae"] = torch.mean(loss_mae)
            loss[ratio]["loss_huber"] = torch.mean(loss_huber)

            if self.args.use_tdalign:
                # DeepEDM uses TDAlign loss
                d_true = torch.diff(true[i], dim=1)
                d_pred = torch.diff(pred[i], dim=1)
                loss_sgn = torch.sum(
                    torch.mul(d_true, d_pred) < 0, dim=1
                ) / d_true.size(1)
                if self.args.loss == "mse":
                    loss_d = torch.mean((d_true - d_pred) ** 2, dim=1)
                    loss_per_variable = torch.mean(
                        loss_sgn * loss_mse + (1 - loss_sgn) * loss_d,
                        dim=0,
                    )
                elif self.args.loss == "mae":
                    loss_d = torch.mean(torch.abs(d_true - d_pred), dim=1)
                    loss_per_variable = torch.mean(
                        loss_sgn * loss_mae + (1 - loss_sgn) * loss_d,
                        dim=0,
                    )
                elif self.args.loss == "huber":
                    loss_d = torch.mean(
                        F.huber_loss(
                            d_pred,
                            d_true,
                            reduction="none",
                            delta=self.args.huber_delta,
                        ),
                        dim=1,
                    )
                    loss_per_variable = torch.mean(
                        loss_sgn * loss_huber + (1 - loss_sgn) * loss_d,
                        dim=0,
                    )
                else:
                    raise ValueError(f"Unsupported loss: {self.args.loss}")
            elif self.args.loss == "mse":
                loss_per_variable = torch.mean(loss_mse, dim=0)
            elif self.args.loss == "mae":
                loss_per_variable = torch.mean(loss_mae, dim=0)
            elif self.args.loss == "huber":
                loss_per_variable = torch.mean(loss_huber, dim=0)
            else:
                raise ValueError(f"Unsupported loss: {self.args.loss}")

            loss_opt = loss_opt + loss_per_variable.sum()
            total_elements += loss_per_variable.numel()

        loss["loss_opt"] = loss_opt / total_elements

        return loss

    def _save_epoch_loss(
        self,
        res,
        itr,
        ratio,
        epoch_loss,
        flag=None,
    ):
        if not self.args.save_result:
            return
        results_path = os.path.join(self.args.results, res, itr)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        np.save(
            results_path + "/" + f"{flag}_loss_opt.npy",
            np.array(epoch_loss["loss_opt"]),
        )

        for r in ratio:
            results_path = os.path.join(self.args.results, res, itr, str(r))
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            np.save(
                results_path + "/" + f"{flag}_loss_mse.npy",
                np.array(epoch_loss[r]["loss_mse"]),
            )
            np.save(
                results_path + "/" + f"{flag}_loss_mae.npy",
                np.array(epoch_loss[r]["loss_mae"]),
            )
            np.save(
                results_path + "/" + f"{flag}_loss_huber.npy",
                np.array(epoch_loss[r]["loss_huber"]),
            )

    def _run_other_model(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_y = [y.float().to(self.device, non_blocking=True) for y in batch_y]
        batch_y_mark = [
            y_mark.float().to(self.device, non_blocking=True)
            for y_mark in batch_y_mark
        ]
        batch_x = [x.float().to(self.device, non_blocking=True) for x in batch_x]
        batch_x_mark = [
            x_mark.float().to(self.device, non_blocking=True)
            for x_mark in batch_x_mark
        ]

        # Align all variables to the highest sampling rate using the fixed
        # repeat strategy. data_alignment defaults to method="repeat".
        batch_x = data_alignment(batch_x, self.args.ratio)
        tmp_batch_y = data_alignment(batch_y, self.args.ratio)
        batch_x_mark = data_alignment(batch_x_mark, self.args.ratio)
        batch_y_mark = data_alignment(batch_y_mark, self.args.ratio)

        # decoder input
        dec_inp = torch.zeros_like(tmp_batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = (
            torch.cat([tmp_batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
            .float()
            .to(self.device, non_blocking=True)
        )
        del tmp_batch_y

        # encoder - decoder
        direct_input_models = {"DLinear", "CrossLinear", "SparseTSF", "Duet"}
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.model in direct_input_models:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.model in direct_input_models:
                outputs = self.model(batch_x)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs = outputs[:, -self.args.pred_len :, :]
        pred = []
        true = []
        split_dim = self.args.enc_in // len(self.args.ratio)
        for i, ratio in enumerate(self.args.ratio):
            if i < len(self.args.ratio) - 1:
                start = i * split_dim
                end = (i + 1) * split_dim
            else:
                start = i * split_dim
                end = self.args.enc_in
            data = outputs[:, :, start:end]
            data = data[:, ::ratio, :]
            
            pred.append(data)
            true.append(batch_y[i][:, -(self.args.pred_len // ratio) :, :])
        return pred, true

    def _run_our_model(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
    ):
        batch_y = [
            y[:, -(self.args.pred_len // self.args.ratio[i]) :, :]
            .float()
            .to(self.device, non_blocking=True)
            for i, y in enumerate(batch_y)
        ]
        batch_x = [x.float().to(self.device, non_blocking=True) for x in batch_x]
        batch_x_mark = [
            x_mark.float().to(self.device, non_blocking=True)
            for x_mark in batch_x_mark
        ]

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    batch_x,
                    batch_x_mark,
                )
        else:
            outputs = self.model(
                batch_x,
                batch_x_mark,
            )
        # outputs is list of tensors, each tensor with shape (batch_size, self.args.pred_len // ratio, crossponding enc_in)
        pred = outputs
        true = batch_y
        return pred, true

    def vali(self, vali_data, vali_loader):
        vali_loss = {
            "loss_opt": [],
            **{
                ratio: {
                    "loss_mse": [],
                    "loss_mae": [],
                    "loss_huber": [],
                }
                for ratio in self.args.ratio
            },
        }
        vali_loss_avg = {
            "loss_opt": 0.0,
            **{
                ratio: {
                    "loss_mse": 0.0,
                    "loss_mae": 0.0,
                    "loss_huber": 0.0,
                }
                for ratio in self.args.ratio
            },
        }
        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_instance_idx,
                _,
            ) in enumerate(vali_loader):
                if self.args.model != "TSSI":
                    pred, true = self._run_other_model(
                        batch_x, batch_y, batch_x_mark, batch_y_mark
                    )
                    loss = self._compute_loss(pred, true)
                else:
                    pred, true = self._run_our_model(batch_x, batch_y, batch_x_mark)
                    loss = self._compute_loss(pred, true)

                vali_loss["loss_opt"].append(loss["loss_opt"].item())
                for ratio in self.args.ratio:
                    vali_loss[ratio]["loss_mse"].append(loss[ratio]["loss_mse"].item())
                    vali_loss[ratio]["loss_mae"].append(loss[ratio]["loss_mae"].item())
                    vali_loss[ratio]["loss_huber"].append(
                        loss[ratio]["loss_huber"].item()
                    )

        vali_loss_avg["loss_opt"] = np.average(vali_loss["loss_opt"])
        for ratio in self.args.ratio:
            vali_loss_avg[ratio]["loss_mse"] = np.average(vali_loss[ratio]["loss_mse"])
            vali_loss_avg[ratio]["loss_mae"] = np.average(vali_loss[ratio]["loss_mae"])
            vali_loss_avg[ratio]["loss_huber"] = np.average(
                vali_loss[ratio]["loss_huber"]
            )
        self.model.train()
        return vali_loss_avg

    def train(self, res, itr):
        train_data, train_loader = self._get_data(flag="train", ratio=self.args.ratio)
        vali_data, vali_loader = self._get_data(flag="vali", ratio=self.args.ratio)
        test_data, test_loader = self._get_data(flag="test", ratio=self.args.ratio)

        if self.args.save_epoch_loss:
            e_train_loss = {
                "loss_opt": [],
                **{
                    ratio: {
                        "loss_mse": [],
                        "loss_mae": [],
                        "loss_huber": [],
                    }
                    for ratio in self.args.ratio
                },
            }
            e_vali_loss = {
                "loss_opt": [],
                **{
                    ratio: {
                        "loss_mse": [],
                        "loss_mae": [],
                        "loss_huber": [],
                    }
                    for ratio in self.args.ratio
                },
            }
            e_test_loss = {
                "loss_opt": [],
                **{
                    ratio: {
                        "loss_mse": [],
                        "loss_mae": [],
                        "loss_huber": [],
                    }
                    for ratio in self.args.ratio
                },
            }

        ratio_str = "_".join([str(ratio) for ratio in self.args.ratio])
        model_path = os.path.join(self.args.checkpoints, res, itr, ratio_str)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.model == "SparseTSF" and self.args.lradj == "TST":
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate,
            )
        else:
            scheduler = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = {
                "loss_opt": [],
                **{
                    ratio: {
                        "loss_mse": [],
                        "loss_mae": [],
                        "loss_huber": [],
                    }
                    for ratio in self.args.ratio
                },
            }
            train_loss_avg = {
                "loss_opt": 0.0,
                **{
                    ratio: {
                        "loss_mse": 0.0,
                        "loss_mae": 0.0,
                        "loss_huber": 0.0,
                    }
                    for ratio in self.args.ratio
                },
            }

            self.model.train()
            epoch_time = time.time()

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_instance_idx,
                _,
            ) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.model != "TSSI":
                    pred, true = self._run_other_model(
                        batch_x, batch_y, batch_x_mark, batch_y_mark
                    )
                    loss = self._compute_loss(pred, true)
                else:
                    pred, true = self._run_our_model(batch_x, batch_y, batch_x_mark)
                    loss = self._compute_loss(pred, true)

                train_loss["loss_opt"].append(loss["loss_opt"].item())
                for ratio in self.args.ratio:
                    train_loss[ratio]["loss_mse"].append(loss[ratio]["loss_mse"].item())
                    train_loss[ratio]["loss_mae"].append(loss[ratio]["loss_mae"].item())
                    train_loss[ratio]["loss_huber"].append(
                        loss[ratio]["loss_huber"].item()
                    )

                if self.args.use_amp:
                    scaler.scale(loss["loss_opt"]).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss["loss_opt"].backward()
                    if self.args.model == "DeepEDM":
                        if self.args.clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.clip_grad_norm
                            )
                    model_optim.step()

                if scheduler is not None:
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.args, printout=False
                    )
                    scheduler.step()

            print(
                "Epoch: {0}, Steps: {1}, Cost Time: {2}".format(
                    epoch + 1, train_steps, time.time() - epoch_time
                )
            )

            train_loss_avg["loss_opt"] = np.average(train_loss["loss_opt"])
            for ratio in self.args.ratio:
                train_loss_avg[ratio]["loss_mse"] = np.average(
                    train_loss[ratio]["loss_mse"]
                )
                train_loss_avg[ratio]["loss_mae"] = np.average(
                    train_loss[ratio]["loss_mae"]
                )
                train_loss_avg[ratio]["loss_huber"] = np.average(
                    train_loss[ratio]["loss_huber"]
                )

            vali_loss_avg = self.vali(vali_data, vali_loader)
            test_loss_avg = self.vali(test_data, test_loader)
            print(
                "train_loss_opt:{0:.7f} | {1}".format(
                    train_loss_avg["loss_opt"],
                    " | ".join(
                        [
                            f"train_{ratio}_mse:{train_loss_avg[ratio]['loss_mse']:.7f} | train_{ratio}_mae:{train_loss_avg[ratio]['loss_mae']:.7f} | train_{ratio}_huber:{train_loss_avg[ratio]['loss_huber']:.7f}"
                            for ratio in self.args.ratio
                        ]
                    ),
                )
            )
            print(
                "vali_loss_opt:{0:.7f} | {1}".format(
                    vali_loss_avg["loss_opt"],
                    " | ".join(
                        [
                            f"vali_{ratio}_mse:{vali_loss_avg[ratio]['loss_mse']:.7f} | vali_{ratio}_mae:{vali_loss_avg[ratio]['loss_mae']:.7f} | vali_{ratio}_huber:{vali_loss_avg[ratio]['loss_huber']:.7f}"
                            for ratio in self.args.ratio
                        ]
                    ),
                )
            )
            print(
                "test_loss_opt:{0:.7f} | {1}".format(
                    test_loss_avg["loss_opt"],
                    " | ".join(
                        [
                            f"test_{ratio}_mse:{test_loss_avg[ratio]['loss_mse']:.7f} | test_{ratio}_mae:{test_loss_avg[ratio]['loss_mae']:.7f} | test_{ratio}_huber:{test_loss_avg[ratio]['loss_huber']:.7f}"
                            for ratio in self.args.ratio
                        ]
                    ),
                )
            )

            if self.args.save_epoch_loss:
                e_train_loss["loss_opt"].append(train_loss_avg["loss_opt"])
                for ratio in self.args.ratio:
                    e_train_loss[ratio]["loss_mse"].append(
                        train_loss_avg[ratio]["loss_mse"]
                    )
                    e_train_loss[ratio]["loss_mae"].append(
                        train_loss_avg[ratio]["loss_mae"]
                    )
                    e_train_loss[ratio]["loss_huber"].append(
                        train_loss_avg[ratio]["loss_huber"]
                    )

                e_vali_loss["loss_opt"].append(vali_loss_avg["loss_opt"])
                for ratio in self.args.ratio:
                    e_vali_loss[ratio]["loss_mse"].append(
                        vali_loss_avg[ratio]["loss_mse"]
                    )
                    e_vali_loss[ratio]["loss_mae"].append(
                        vali_loss_avg[ratio]["loss_mae"]
                    )
                    e_vali_loss[ratio]["loss_huber"].append(
                        vali_loss_avg[ratio]["loss_huber"]
                    )

                e_test_loss["loss_opt"].append(test_loss_avg["loss_opt"])
                for ratio in self.args.ratio:
                    e_test_loss[ratio]["loss_mse"].append(
                        test_loss_avg[ratio]["loss_mse"]
                    )
                    e_test_loss[ratio]["loss_mae"].append(
                        test_loss_avg[ratio]["loss_mae"]
                    )
                    e_test_loss[ratio]["loss_huber"].append(
                        test_loss_avg[ratio]["loss_huber"]
                    )

            early_stopping(vali_loss_avg["loss_opt"], self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping!!!")
                break

            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = model_path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))

        if self.args.save_epoch_loss:
            self._save_epoch_loss(
                res,
                itr,
                self.args.ratio,
                e_train_loss,
                flag="train",
            )
            self._save_epoch_loss(
                res,
                itr,
                self.args.ratio,
                e_vali_loss,
                flag="vali",
            )
            self._save_epoch_loss(
                res,
                itr,
                self.args.ratio,
                e_test_loss,
                flag="test",
            )

        return self.model

    def test(self, res, itr):
        test_data, test_loader = self._get_data(flag="test", ratio=self.args.ratio)

        preds = [[] for _ in range(len(self.args.ratio))]
        trues = [[] for _ in range(len(self.args.ratio))]
        inputxs = [[] for _ in range(len(self.args.ratio))]
        x_marks = [[] for _ in range(len(self.args.ratio))]
        y_marks = [[] for _ in range(len(self.args.ratio))]
        # in lower rate data, some instances are reused, so we use instance_idx to identify the unique instances for each rate
        instance_idxs = [[] for _ in range(len(self.args.ratio))]

        mses = []
        maes = []
        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_instance_idx,
                _,
            ) in enumerate(test_loader):
                if self.args.model != "TSSI":
                    pred, true = self._run_other_model(
                        batch_x, batch_y, batch_x_mark, batch_y_mark
                    )
                else:
                    pred, true = self._run_our_model(batch_x, batch_y, batch_x_mark)

                for i in range(len(self.args.ratio)):
                    preds[i].append(pred[i].detach().cpu().numpy())
                    trues[i].append(true[i].detach().cpu().numpy())
                    inputxs[i].append(batch_x[i].detach().cpu().numpy())
                    x_marks[i].append(batch_x_mark[i].detach().cpu().numpy())
                    y_marks[i].append(
                        batch_y_mark[i][:, -pred[i].shape[1] :].detach().cpu().numpy()
                    )
                    instance_idxs[i].append(
                        batch_instance_idx[i].detach().cpu().numpy()
                    )

        del (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            batch_instance_idx,
            pred,
            true,
        )
        preds = [np.concatenate(p, axis=0) for p in preds]
        trues = [np.concatenate(t, axis=0) for t in trues]
        inputxs = [np.concatenate(i, axis=0) for i in inputxs]
        x_marks = [np.concatenate(xm, axis=0) for xm in x_marks]
        y_marks = [np.concatenate(ym, axis=0) for ym in y_marks]
        instance_idxs = [np.concatenate(a, axis=0) for a in instance_idxs]

        # result save
        for i, r in enumerate(self.args.ratio):
            print(
                f"ratio:{r}, test, unique instances:{len(np.unique(instance_idxs[i]))}"
            )
            mae, mse, rmse, mape, mspe, rse, corr = instance_averaged_metrics(
                preds[i], trues[i], instance_idxs[i]
            )
            print(
                "ratio:{}, mse:{}, mae:{}".format(
                    r,
                    mse,
                    mae,
                )
            )
            mses.append(mse)
            maes.append(mae)

            if self.args.save_result:
                # common results path: trues, inputxs, instance_idxs
                save_dir_split = self.args.results.split("/")
                lengths = save_dir_split[-1]
                ratio_str = save_dir_split[-2]
                data_name = save_dir_split[-4]
                save_dir = save_dir_split[:-4]
                save_dir = "/".join(save_dir)
                com_results_path = os.path.join(
                    save_dir, "com", data_name, ratio_str, res, lengths, str(r)
                )
                if not os.path.exists(com_results_path):
                    os.makedirs(com_results_path)
                    np.save(com_results_path + "/" + "trues.npy", trues[i])
                    np.save(com_results_path + "/" + "inputxs.npy", inputxs[i])
                    np.save(com_results_path + "/" + "x_marks.npy", x_marks[i])
                    np.save(com_results_path + "/" + "y_marks.npy", y_marks[i])
                    np.save(
                        com_results_path + "/" + "instance_idxs.npy",
                        instance_idxs[i],
                    )

                # specific results path: metrics, preds
                spe_results_path = os.path.join(self.args.results, res, itr, str(r))
                if not os.path.exists(spe_results_path):
                    os.makedirs(spe_results_path)
                np.save(
                    spe_results_path + "/" + "metrics.npy",
                    np.array([mae, mse, rmse, mape, mspe, rse]),
                )
                np.save(spe_results_path + "/" + "preds.npy", preds[i])
        del preds, trues, inputxs, instance_idxs
        return mses, maes

    def predict(self, res, itr):
        test_data, test_loader = self._get_data(flag="test", ratio=self.args.ratio)

        ratio_str = "_".join([str(ratio) for ratio in self.args.ratio])
        model_path = os.path.join(self.args.checkpoints, res, itr, ratio_str)
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")
        best_model_path = model_path + "/" + "checkpoint.pth"
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=torch.device("cuda"))
        )

        preds = [[] for _ in range(len(self.args.ratio))]
        trues = [[] for _ in range(len(self.args.ratio))]
        inputxs = [[] for _ in range(len(self.args.ratio))]
        x_marks = [[] for _ in range(len(self.args.ratio))]
        y_marks = [[] for _ in range(len(self.args.ratio))]
        # in lower rate data, some instances are reused, so we use instance_idx to identify the unique instances for each rate
        instance_idxs = [[] for _ in range(len(self.args.ratio))]

        mses = []
        maes = []

        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_instance_idx,
                _,
            ) in enumerate(test_loader):
                if self.args.model != "TSSI":
                    pred, true = self._run_other_model(
                        batch_x, batch_y, batch_x_mark, batch_y_mark
                    )
                else:
                    pred, true = self._run_our_model(batch_x, batch_y, batch_x_mark)

                for i in range(len(self.args.ratio)):
                    preds[i].append(pred[i].detach().cpu().numpy())
                    trues[i].append(true[i].detach().cpu().numpy())
                    inputxs[i].append(batch_x[i].detach().cpu().numpy())
                    x_marks[i].append(batch_x_mark[i].detach().cpu().numpy())
                    y_marks[i].append(
                        batch_y_mark[i][:, -pred[i].shape[1] :].detach().cpu().numpy()
                    )
                    instance_idxs[i].append(
                        batch_instance_idx[i].detach().cpu().numpy()
                    )

        del (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            batch_instance_idx,
            pred,
            true,
        )
        preds = [np.concatenate(p, axis=0) for p in preds]
        trues = [np.concatenate(t, axis=0) for t in trues]
        inputxs = [np.concatenate(i, axis=0) for i in inputxs]
        x_marks = [np.concatenate(xm, axis=0) for xm in x_marks]
        y_marks = [np.concatenate(ym, axis=0) for ym in y_marks]
        instance_idxs = [np.concatenate(a, axis=0) for a in instance_idxs]

        # result save
        for i, r in enumerate(self.args.ratio):
            print(
                f"ratio:{r}, test, unique instances:{len(np.unique(instance_idxs[i]))}"
            )
            mae, mse, rmse, mape, mspe, rse, corr = instance_averaged_metrics(
                preds[i], trues[i], instance_idxs[i]
            )
            print("ratio:{}, mse:{}, mae:{}".format(r, mse, mae))
            mses.append(mse)
            maes.append(mae)

            if not self.args.save_result:
                continue

            # common results path: trues, inputxs, instance_idxs
            save_dir_split = self.args.results.split("/")
            lengths = save_dir_split[-1]
            ratio_str = save_dir_split[-2]
            data_name = save_dir_split[-4]
            save_dir = save_dir_split[:-4]
            save_dir = "/".join(save_dir)
            # 这里lengths和_未定义，保留原逻辑
            com_results_path = os.path.join(
                save_dir, "com", data_name, ratio_str, res, lengths, str(r)
            )
            if not os.path.exists(com_results_path):
                os.makedirs(com_results_path)
                np.save(com_results_path + "/" + "trues.npy", trues[i])
                np.save(com_results_path + "/" + "inputxs.npy", inputxs[i])
                np.save(com_results_path + "/" + "x_marks.npy", x_marks[i])
                np.save(com_results_path + "/" + "y_marks.npy", y_marks[i])
                np.save(com_results_path + "/" + "instance_idxs.npy", instance_idxs[i])
            else:
                pass
            # specific results path: metrics, preds
            spe_results_path = os.path.join(self.args.results, res, itr, str(r))
            if not os.path.exists(spe_results_path):
                os.makedirs(spe_results_path)

            np.save(
                spe_results_path + "/" + "metrics.npy",
                np.array([mae, mse, rmse, mape, mspe, rse]),
            )
            np.save(spe_results_path + "/" + "preds.npy", preds[i])
        del preds, trues, inputxs, instance_idxs
        return mses, maes
