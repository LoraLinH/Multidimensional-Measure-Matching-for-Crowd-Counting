import torch
from torch import nn
import numpy as np
from math import acos

scale=2.0*0.6/acos(1e-5)

class Post_Prob_Scale(nn.Module):
    def __init__(self, scale_list, c_size, stride, device):
        super(Post_Prob_Scale, self).__init__()
        assert c_size % stride == 0

        self.device = device
        self.cood_s = torch.tensor(sorted(scale_list), dtype=torch.float32, device=device)
        self.cood_s.unsqueeze_(0) # 1 * level
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride // 2
        self.cood.unsqueeze_(0)  # 1 * cood
        self.c_size = c_size

    def forward(self, points):
        x = points[:, 0].unsqueeze_(1)
        y = points[:, 1].unsqueeze_(1)
        s = points[:, 2].unsqueeze_(1)

        x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood  # N * Cood_X
        y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood  # N * Cood_Y
        s_dis = (self.cood_s - s) **2
        s_dis = torch.log(s_dis + 1e-5) / (self.c_size**2)
        y_dis.unsqueeze_(2)
        x_dis.unsqueeze_(1)
        dis = y_dis + x_dis  # N * Cood_Y * Cood_X
        dis = dis.view((dis.size(0), -1)) / self.c_size**2  # N * Cood_All
        dis = dis.unsqueeze_(1) + s_dis.unsqueeze_(2)  # level * N * cood_all
        dis = dis.view((dis.size(0), -1))
        dis = torch.sqrt(torch.relu(dis))

        return dis
