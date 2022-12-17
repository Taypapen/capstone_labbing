import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from fastai.layers import *
from torchvision import models


def noop(x):
    return x


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths):
    return torch.stack([torch.mean( x[index][:,0:i], 1) for index, i in enumerate(lengths)],0)


def _3d_block(in_size, out_size, kernel_size, stride, padding, bias=False, relu_type='prelu'):
    return nn.Sequential(
        nn.Conv3d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm3d(out_size),
        nn.PReLU(num_parameters=out_size) if relu_type== 'prelu' else nn.ReLU()
    )


def _downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )


def _bottleneck_conv_block(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni,nf//4, 1),
        ConvLayer(nf//4,nf//4, stride=stride),
        ConvLayer(nf//4, nf, 1, act_cls=None, norm_type=NormType.BatchZero)
    )


def _conv_block(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni, nf, stride=stride),
        ConvLayer(nf, nf, 1, act_cls=None, norm_type=NormType.BatchZero)
    )


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, bottled=False):
        super(ResBlock, self).__init__()
        print(ni)
        self.convs = _bottleneck_conv_block(ni, nf, stride) if bottled else _conv_block(ni, nf, stride)
        self.idconv = noop if ni == nf else ConvLayer(ni, nf, 1, act_cls=None)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


class ResNet(nn.Sequential):
    def __init__(self, layers, expansion=1):

        #self.relu_type= relu_type
        self.bottled = True if expansion > 1 else False
        self.block_sizes = [64, 64, 128, 256, 128]
        for i in range(1, len(self.block_sizes)): self.block_sizes[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]

        super().__init__(*blocks, nn.AdaptiveAvgPool2d(1))

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx==0 else 2
        ch_in, ch_out = self.block_sizes[idx:idx+2]
        return nn.Sequential(*[
            ResBlock(ch_in if i==0 else ch_out, ch_out, stride if i==0 else 1, self.bottled)
            for i in range(n_layers)
        ])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()


class BasicBlock1D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, relu_type='relu'):
        super(BasicBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm1d(n_outputs)
        self.chomp1 = Chomp1d(padding, True)
        if relu_type == 'relu':
            self.relu1 = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm1d(n_outputs)
        self.chomp2 = Chomp1d(padding, True)
        if relu_type == 'relu':
            self.relu2 = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu2 = nn.PReLU(num_parameters=n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if relu_type == 'relu':
            self.relu = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ConvNet1D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, relu_type='relu'):
        super(ConvNet1D, self).__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i==0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(BasicBlock1D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size-1) * dilation_size, dropout=dropout, relu_type=relu_type))
        self.network = nn.Sequential(*layers)

    def forward(self, x, lengths):
        x = self.network(x)
        x = _average_batch(x, lengths)
        return x
