import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F
from fastai.layers import *


def noop(x):
    return x


def _conv_block(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni, nf, stride=stride),
        ConvLayer(nf, nf, 1, act_cls=None, norm_type=NormType.BatchZero)
    )


def _bottleneck_conv_block(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni, nf//4, 1),
        ConvLayer(nf//4, nf//4, stride=stride),
        ConvLayer(nf//4, nf, 1, act_cls=None, norm_type=NormType.BatchZero)
    )


def _resnet_stem(*sizes):
    return [
        ConvLayer(sizes[i], sizes[i+1], 3, stride=2 if i == 0 else 1)
        for i in range(len(sizes)-1)
    ] + [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super(ResBlock, self).__init__()
        self.convs = _bottleneck_conv_block(ni, nf, stride)
        self.idconv = noop if ni == nf else ConvLayer(ni, nf, 1, act_cls=None)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


class CustomResNet(nn.Sequential):
    def __init__(self, layers, expansion=1):

        stem = _resnet_stem(64, 64, 128)
        self.block_sizes = [128, 128, 256, 512]
        #for i in range(1, len(self.block_sizes)): self.block_sizes[i] *= expansion
        for i in range(1, 4): self.block_sizes[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]

        super().__init__(*stem, *blocks, nn.AdaptiveAvgPool2d(1))

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx == 0 else 2
        ch_in, ch_out = self.block_sizes[idx:idx+2]
        return nn.Sequential(*[
            ResBlock(ch_in if i == 0 else ch_out, ch_out, stride if i==0 else 1)
            for i in range(n_layers)
        ])

