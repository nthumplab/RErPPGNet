from __future__ import print_function, division
import torch.nn as nn
from de_rPPG_model import conv3D_module, ResidualBlock
import torch

class RppgAtt_v2(nn.Module):
    def __init__(self, medium_channels, task):
        super(RppgAtt_v2, self).__init__()
        self.task = task # removal or embedd
        self.medium_channels = medium_channels
        self.conv2d1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3+1, # RGB + rPPG signal
                out_channels=medium_channels,
                kernel_size=(1, 5, 5), # v2
                stride=1,
                padding=(0, 2, 2), # v2
            ),
            nn.ReLU(),
            nn.BatchNorm3d(medium_channels),
        )
        self.res1 = ResidualBlock(medium_channels, medium_channels, stride=1, downsample=None, last_block = False)
        self.res2 = ResidualBlock(medium_channels, medium_channels, stride=1, downsample=None, last_block=False)
        self.conv_final = conv3D_module(medium_channels, medium_channels, last_layer = False)

        self.attention_conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=medium_channels,
                out_channels=1, # Gray
                kernel_size=(3, 5, 5), # v2
                stride=1,
                padding=(1, 2, 2), # v2
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(1), # out_channel
            nn.Sigmoid(),
        )
        
        self.signal_conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=medium_channels,
                out_channels=3, # RGB
                kernel_size=(3, 5, 5), # v2
                stride=1,
                padding=(1, 2, 2), # v2
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(3), # out_channel
            nn.Tanh(),
        )

    def forward(self, face, rppg):
        if self.task == "embedd":
            rppg = rppg.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # Step 1: Concate video and signal at channel-dim
        rppg_feat = rppg.repeat(1,1,1,face.size(3),face.size(4))
        input_image = torch.cat((face, rppg_feat), 1)

        # Step 2: Basic model
        feat = self.conv2d1(input_image)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.conv_final(feat)

        # Step 3: Get attention map and signal video
        att_map = self.attention_conv3d(feat)
        sig_video = self.signal_conv3d(feat)

        # Step 4: Fuse input video and signal video
        out = att_map * face + (1 - att_map) * sig_video
        
        return out, att_map, sig_video