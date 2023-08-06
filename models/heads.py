

from torch import nn

from .common_layers import get_act_layer

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, dims, num_classes, act="softmax"):
        super(ClassificationHead, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.act = get_act_layer(act)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        m = []
        for i in range(len(dims)):
            m.append(nn.Conv2d(in_dim, dims[i], 1, 1))
            m.append(nn.BatchNorm2d(dims[i]))
            m.append(get_act_layer("relu"))
            in_dim = dims[i]
        m.append(nn.Conv2d(in_dim, self.num_classes, 1, 1))
        self.fcs = nn.Sequential(*m)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.fcs(x)
        x = x.view(x.size(0), -1)
        if self.training:
            return x
        else:
            x_prob = self.act(x)
            return x, x_prob