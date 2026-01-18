import numpy as np
import pandas as pd
from gan1 import Generator, Discriminator
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from einops import rearrange
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from utils import normalization
import math
from torch.utils.tensorboard import SummaryWriter
from residual_loss import residual_loss_fn
from data import load_dataset

class MyDataset(Dataset):
    def __init__(self, p, m, l):
        self.power = p
        self.meteo = m
        self.load = l

    def __len__(self):
        return len(self.power)

    def __getitem__(self, idx):
        power = self.power[idx]
        meteo = self.meteo[idx]
        load = self.load[idx]

        return power, meteo, load

class model1():
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gen = Generator(self.config).to(self.device)
        self.dis = Discriminator().to(self.device)
        self.logger = SummaryWriter(log_dir='log')



    def train_step(self, p, m, l):
        # d_optim = torch.optim.Adam(self.dis.parameters(), lr=0.001)
        g_optim = torch.optim.AdamW(self.gen.parameters(), lr=0.0015)
        loss_fn = torch.nn.MSELoss()



        # 对数据集进行迭代
        pe = torch.zeros(576, 1)
        position = torch.arange(0, 576, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 1, 2).float() * (-math.log(10000.0) / 1))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.squeeze()
        pe = pe.repeat(16).reshape((16, 1, 576))
        pe = pe.to(self.device).float()

        p = p.to(self.device).float()  # b 1 t
        m = m.to(self.device).float()
        l = l.to(self.device).float()
        # pre为tensor.gpu
        # numpy_pre为numpy.cpu


        # l = l.cpu().numpy()
        p_his = p[:, :, :-16] / 500
        p_true = p[:, :, -16:]


        m_his = m[:, :, :-16]
        m_true = m[:, :, -16:]

        l_true = l[:, :, -16:]

        input = torch.concat((p_his, m_his, pe), dim=1)

        #b c l 576

        # 生成器训练
        g_optim.zero_grad()  # 先将生成器上的梯度置零
        fake_pred, res, rec_loss = self.gen(input, l_true, m_true, p_true)
        fake_pred = fake_pred.squeeze()
        p_true = p_true.squeeze()
        # fake_output = self.dis(input, fake_sol, l_true)
        # g_loss = loss_fn(fake_sol,
        #                  sol_true
        #                  )  # 生成器损失
        g_loss = loss_fn(fake_pred, p_true)  # 生成器损失
        res_loss = residual_loss_fn(res)
        g_loss_sum = g_loss + res_loss
        # + 0.001*rec_loss
        g_loss_sum.backward()

        g_optim.step()

        return g_loss, res_loss, rec_loss


    def train(self, epoch):
        dataset = load_dataset()
        for i in range(epoch):
            g_epoch = 0
            r_epoch = 0
            for step, data in enumerate(dataset):
                # k = k + 1
                p, m, l = data
                g, r, dispatch = self.train_step(p, m, l)
                with torch.no_grad():
                    g_epoch += g
                    r_epoch += r
            g_epoch /= 195
            r_epoch /= 195
            print("step: {:<6d} \t G: {:<.3f} \t R: {:<.3f}".format(i+1, g_epoch, r_epoch))
            self.logger.add_scalar('gen_loss', g_epoch, i + 1)
            self.logger.add_scalar('res_loss', r_epoch, i + 1)


        # torch.save(self.gen.state_dict(), 'patchtst.pth')
        torch.save(self.dis.state_dict(), 'onlymsd.pth')



    

    def loadWeights(self):
        gen = Generator(self.config)

        # state_dict = model.state_dict()
        # weights = torch.load(weights_path)['model_state_dict']  # 读取预训练模型权重
        # model.load_state_dict(weights)

        # 加载预训练模型
        g_dict = torch.load("gen.pth")
        gen.load_state_dict(g_dict, False)
