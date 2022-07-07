from __future__ import print_function, division
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_conv = 2, pool = False):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_conv = num_conv
        layers = []
        channels = [in_channel] + [out_channel for i in range(num_conv)]
        for i in range(len(channels) - 1):
            if pool:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
            layers.append(
                nn.Conv3d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, padding=1, padding_mode='replicate', bias=True))
            layers.append(nn.BatchNorm3d(num_features=channels[i + 1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())

        self.op = nn.Sequential(*layers)

    def forward(self, x):
        activation = self.op(x)
        return activation