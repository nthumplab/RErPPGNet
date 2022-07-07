from __future__ import print_function, division
import torch.nn as nn

class conv3D_module(nn.Module):
    def __init__(self, in_channel, out_channel, last_layer=False):
        super(conv3D_module, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if last_layer == False:
            self.conv3d1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=(3, 3, 3),
                    stride=1,
                    padding=(1, 1, 1),
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),  # activation
            )
        else:
            self.conv3d1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=(3, 3, 3),
                    stride=1,
                    padding=(1, 1, 1),
                ),
                nn.BatchNorm3d(out_channel),
                nn.Tanh(),  # activation
            )

    def forward(self, x):
        activation = self.conv3d1(x)
        return activation

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, last_block = False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = conv3D_module(in_channels, out_channels, last_layer = False)
        self.conv2 = conv3D_module(out_channels, out_channels, last_layer = last_block)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = residual + out
        out = self.relu(out)
        return out
