from torch import nn
import torch.nn.functional as F
import torch
from models.pvt_v2 import pvt_v2_b3


class VGG_PVT(nn.Module):
    def __init__(self):
        super(VGG_PVT, self).__init__()
        self.features = pvt_v2_b3()


        self.trans_2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True))

        self.reg_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1),
            nn.ReLU(inplace=True))

        self.smooth_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.reg_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

        self.trans_0 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True))

        self.smooth_0 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.reg_0 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        '''
        self.trans_9 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True))

        self.smooth_9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.reg_9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        '''

    def _upsample_add(self, x, y, scale):
        _, _, h, w = y.size()
        return F.upsample(x, size=(h, w), mode='bilinear') / scale**2 + y

    def forward(self, x):
        x9, x0, x1, x2 = self.features(x)

        out_list = []
        x2 = self.trans_2(x2)
        x2_out = torch.abs(self.reg_2(x2))
        out_list.append(x2_out)

        x1 = self.trans_1(x1)
        x1 = self._upsample_add(x2, x1, 2)
        x1 = self.smooth_1(x1)
        x1_out = torch.abs(self.reg_1(x1))
        out_list.append(x1_out)

        x0 = self.trans_0(x0)
        x0 = self._upsample_add(x1, x0, 2)
        x0 = self.smooth_0(x0)
        x0_out = torch.abs(self.reg_0(x0))
        out_list.append(x0_out)

        # x9 = self.trans_9(x9)
        # x9 = self._upsample_add(x0, x9, 2)
        # x9 = self.smooth_9(x9)
        # x9_out = torch.abs(self.reg_9(x9))
        # out_list.append(x9_out)


        out_list = out_list[::-1]
        _, _, h, w = out_list[1].size()
        out_list[2] = F.upsample(out_list[2], size=(h, w), mode='bilinear') / 4 ** 1
        out_list[0] = F.interpolate(out_list[0], size=(h, w), mode='bilinear') * 4
        # _, _, h, w = out_list[2].size()
        # out_list[3] = F.upsample(out_list[3], size=(h, w), mode='bilinear') / 4
        # out_list[1] = F.interpolate(out_list[1], size=(h, w), mode='bilinear') * 4
        # out_list[0] = F.interpolate(out_list[0], size=(h, w), mode='bilinear') * 16

        out = torch.cat(out_list, dim=1)

        return out


def make_layers_1(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256],
    'B': ['M', 512, 512, 512, 512],
    'C': ['M', 512, 512, 512, 512]
}


def vgg19_pvt():
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    pretrained = True
    model = VGG_PVT()
    if pretrained:
        model.features.load_state_dict(torch.load('models/pvt_v2_b3.pth'), strict=False)
    return model
