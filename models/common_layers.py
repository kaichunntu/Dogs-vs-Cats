
import torch
from torch import nn


def get_act_layer(act):
    if act=="relu":
        return nn.ReLU()
    elif act=="leaky_relu":
        return nn.LeakyReLU()
    elif act=="sigmoid":
        return nn.Sigmoid()
    elif act=="tanh":
        return nn.Tanh()
    elif act=="gelu":
        return nn.GELU()
    elif act=="prelu":
        return nn.PReLU()
    elif act==None:
        return nn.Identity()
    elif act=="softmax":
        return nn.Softmax(-1)
    else:
        raise NotImplementedError("%s" % act)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, act="prelu"):
        super(BaseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = False

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = get_act_layer(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class Stem(nn.Module):
    def __init__(self, in_c, out_c, act="prelu"):
        super(Stem, self).__init__()
        dim = out_c//2
        self.conv1 = BaseConv(in_c, dim, 5, stride=2, padding=2, act=act)
        self.conv2 = BaseConv(dim, dim, 1, stride=1, act=act)
        self.conv3 = BaseConv(dim, out_c, 3, stride=1, padding=1, act=act)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        return x


class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)