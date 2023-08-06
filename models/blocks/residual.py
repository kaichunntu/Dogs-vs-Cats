

from torch import nn
from models.common_layers import BaseConv, get_act_layer

class ResidualBlock(nn.Module):
    def __init__(self, dim, act="relu"):
        super(ResidualBlock, self).__init__()
        self.dim = dim


    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act(out)

        return out
    

class ResNeXtBlock(nn.Module):
    def __init__(self, dim, expand_ratio, c=32, s=1, act="prelu"):
        super(ResNeXtBlock, self).__init__()
        self.dim = dim
        self.expand_ratio = expand_ratio
        self.stride = s>1
        self.conv1 = BaseConv(self.dim, int(self.dim * self.expand_ratio), 1, 1, act=act)
        self.conv2 = BaseConv(int(self.dim * self.expand_ratio), self.dim, kernel_size=3, stride=s, padding=1, groups=c, act=act)
        self.conv3 = BaseConv(self.dim, self.dim, 1, 1, act=None)
        self.act = get_act_layer(act)
        if self.stride:
            self.avg_pool = nn.AvgPool2d((s,s), (s,s))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride:
            residual = self.avg_pool(residual)
        out += residual
        out = self.act(out)

        return out