import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F
from fastai.layers import *
#%%
'''Basic Structure for Neural Network:
    3D Convolution Blocks -> MaxPool3D -> (Convert to 2D tensor) -> ResNet -> (Flatten) -> 1D Convolution Blocks -> Prediction'''
#%%
def noop(x):
    return x
#%%
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)
#%%
def _3d_block(in_size, out_size, kernel_size, stride, padding, bias=False, relu_type='prelu'):
    return nn.Sequential(
        nn.Conv3d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm3d(out_size),
        nn.PReLU(num_parameters=out_size) if relu_type== 'prelu' else nn.ReLU()
    )
#%%
def _bottleneck_conv_block(ni, nf, stride):
    return nn.Sequential(
        ConvLayer(ni,nf//4, 1),
        ConvLayer(nf//4,nf//4, stride=stride),
        ConvLayer(nf//4, nf, 1, act_cls=None, norm_type=NormType.BatchZero)
    )
#%%
def _conv_block(ni,nf,stride):
    return nn.Sequential(
        ConvLayer(ni,nf, stride=stride),
        ConvLayer(nf, nf, 1, act_cls=None, norm_type=NormType.BatchZero)
    )
#%%
class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, bottled=True):
        super(ResBlock, self).__init__()
        self.convs = _bottleneck_conv_block(ni, nf, stride) if bottled else _conv_block(ni, nf, stride)
        self.idconv = noop if ni == nf else ConvLayer(ni, nf, 1, act_cls=None)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self,x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))
#%%
class ResNet(nn.Sequential):
    def __init__(self, layers, expansion=1):

        #self.relu_type= relu_type
        self.block_sizes = [64, 64, 128, 256, 512]
        for i in range(1, len(self.block_sizes)): self.block_sizes[i] *= expansion
        blocks = [self._make_layer(*o) for o in enumerate(layers)]

        super().__init__(*blocks, nn.AdaptiveAvgPool2d(1))
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, idx, n_layers):
        stride = 1 if idx==0 else 2
        ch_in, ch_out = self.block_sizes[idx:idx+2]
        return nn.Sequential(*[
            ResBlock(ch_in if i==0 else ch_out, ch_out, stride if i==0 else 1)
            for i in range(n_layers)
        ])

#%%
class BasicBlock1D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, relu_type='relu'):
        super(BasicBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm1d(n_outputs)
        if relu_type == 'relu':
            self.relu1 = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm1d(n_outputs)
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
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
#%%
class ConvNet1D(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=3, dropout=0.2, relu_type='relu'):
        super(ConvNet1D, self).__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i==0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(BasicBlock1D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=1, dropout=dropout, relu_type=relu_type))
        self.network = nn.Sequential(*layers)
        self.convnet_output = nn.Linear(num_channels[-1], num_classes)
    def forward(self, x):
        x = self.network(x)
        return self.convnet_output(x)
#%%
class Lipreading1(nn.Module):
    def __init__(self, num_classes, relu_type = 'prelu'):
        super(Lipreading1, self).__init__()
        self.kernel_size=3
        self.dropout=0.2
        self.frontend_out = 64
        self.backend_out = 512

        self.frontend3D = _3d_block(1, self.frontend_out, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.max_pool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.trunk = ResNet([3,4,6,3], expansion=4)
        self.flatten = nn.Flatten()
        self.tcn = ConvNet1D(self.backend_out, [256, 512], num_classes, kernel_size=self.kernel_size, dropout=self.dropout, relu_type=relu_type)

        self._initialize_weights_randomly()

    def forward(self, x):
        x = self.frontend3D(x)
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = self.flatten(x)
        return self.tcn(x)

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
#%%
