import torch.nn as nn
from msd_mixer import MSDMixer
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from utils import get_loss_fn
import torch
from einops import rearrange
import numpy as np


class Generator(nn.Module):
    def __init__(self, config):
        # self.out_chn = config.out_chn
        # self.lambda_mse = config.lambda_mse
        # self.lambda_acf = config.lambda_acf
        # self.acf_cutoff = config.acf_cutoff
        # self.optim = config.optim
        # self.lr = config.lr
        # self.lr_factor = config.lr_factor
        # self.weight_decay = config.weight_decay
        # self.patience = config.patience


        super(Generator, self).__init__()
        self.generator = MSDMixer(config.seq_len, config.pred_len, config.in_chn,
                              config.ex_chn, config.out_chn, config.patch_sizes,
                              config.hid_len, config.hid_chn, config.hid_pch,
                              config.hid_pred, config.norm, config.use_last_norm,
                              config.activ, config.drop)

        # self.val_mse = MeanSquaredError()
        # self.val_mae = MeanAbsoluteError()
        # self.test_mse = MeanSquaredError()
        # self.test_mae = MeanAbsoluteError()
        # self.loss_fn = get_loss_fn(config.loss_fn)
        # self.sig = nn.Sigmoid()
        self.act = nn.ReLU()

    def forward(self, x, load, m_true, p_true):
        pred, res, list1, list2 = self.generator(x, m_true)  # 8, 16
        #
        rec = pred - p_true.squeeze()
        # l_rec = self.act(rec[:, 0]).sum()
        l_rec = self.act(rec[0]).sum()

       

        return pred, res, l_rec, list1, list2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.g_l = nn.Sequential(        # blc
            nn.Linear(16, 576),
            nn.LeakyReLU()
        )
        self.g_c = nn.Sequential(
            nn.Linear(7, 9),
            nn.LeakyReLU()
        )
        self.soll = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU()
        )
        self.main1 = nn.Sequential(
            nn.Linear(2, 1),
            nn.LeakyReLU()
        )
        self.main2 = nn.Sequential(
            nn.Linear(576, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.LeakyReLU()
        )
        self.main3 = nn.Linear(9, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, sol, l_true):  # x: b c l     sol: g l b
        sol = self.soll(sol)
        sol1 = torch.concat((sol, l_true), dim=1)
        sol = self.g_l(sol1)
        sol = rearrange(sol, 'b g l -> b l g')
        sol = self.g_c(sol)
        sol = rearrange(sol, 'b l g -> b g l')
        sol = sol.unsqueeze(dim=3)
        x = x.unsqueeze(dim=3)
        input = torch.concat((x, sol), dim=3)  #b g 576 2
        input = self.main1(input).squeeze()
        input = self.main2(input).squeeze()
        output = self.main3(input).squeeze()
        output = self.sig(output)
        return output
