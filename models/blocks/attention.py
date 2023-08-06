
import torch
from torch import nn
from ..common_layers import BaseConv


class Conv2DAttentionBlock(nn.Module):
    def __init__(self, dim, head_k):
        super(Conv2DAttentionBlock, self).__init__()
        self.dim = dim
        self.norm = dim**0.5
        self.head_k = head_k
        expand_dim = dim * head_k

        self.conv_q = nn.Conv2d(dim, expand_dim, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(dim, expand_dim, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(dim, expand_dim, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.transform = BaseConv(expand_dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        
        B, C, H, W = x.shape
        q = self.conv_q(x).reshape(B, self.head_k, self.dim, H*W)
        k = self.conv_k(x).reshape(B, self.head_k, self.dim, H*W)

        score = self.softmax(torch.matmul(q.permute(0, 1, 3, 2), k)/self.norm)

        v = self.conv_v(x).reshape(B, self.head_k, self.dim, H*W)
        weight_sum_v = torch.matmul(score, v.permute(0, 1, 3, 2)).reshape(B, self.head_k*self.dim, H, W)
        x = self.transform(x, weight_sum_v)
        return x
