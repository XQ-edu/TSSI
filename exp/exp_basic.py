import os

import numpy as np
import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                print(f"Use GPUs: {self.args.devices}")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                print(f"Use GPU: {self.args.gpu}")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _load_checkpoint(self, checkpoint_path):
        return torch.load(checkpoint_path, map_location=self.device)

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
