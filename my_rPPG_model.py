from __future__ import print_function, division
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock
from mycbam_v3 import * # Change_cbam

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()
        return

    def forward(self, x, y):
        # for i in range(x.shape[0]):
        vx = x - torch.mean(x, dim = 1, keepdim = True)
        vy = y - torch.mean(y, dim = 1, keepdim = True)
        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        cost = 1 - r
        return cost

class Physnet(nn.Module):
    def __init__(self, seq_length, att_type=None, normalize_attn=True, ):
        super(Physnet, self).__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=8,
                kernel_size=(1,5,5),
                stride=1,
                padding=(0,2,2),
            ),
            nn.ReLU(),
            nn.BatchNorm3d(8),
        )
        self.ST_module1 = ConvBlock(in_channel = 8, out_channel = 16)
        self.ST_module2 = ConvBlock(in_channel = 16, out_channel = 16)
        self.ST_module4 = ConvBlock(in_channel = 16, out_channel = 16)
        self.spatialGlobalAvgpool = nn.AdaptiveAvgPool3d((seq_length,1,1))


        self.att_type = att_type # Change_cbam
        if self.att_type == "MyCBAM_v3":
            self.attn1 = MyCBAM_v3( 16, 16 ) # 前一個16是input feature的channel數量，後者是reduction ratio
            self.attn2 = MyCBAM_v3( 16, 16 )
            self.attn3 = MyCBAM_v3( 16, 16 )
            self.conv2d2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=16,
                    out_channels=1,
                    kernel_size=(1, 1, 1),
                    stride=1,
                    padding=(0, 0, 0),
                )
            )
        elif self.att_type == "without":
            self.conv2d2 = nn.Sequential(
                nn.Conv3d(
                    in_channels=16,
                    out_channels=1,
                    kernel_size=(1, 1, 1),
                    stride=1,
                    padding=(0, 0, 0),
                )
            )
        
    def forward(self, x): 
        N, C, T, W, H = x.size()
        h = self.conv2d1(x)
        l1 = self.ST_module1(h)
        if "MyCBAM" in self.att_type:
            l1, c1 = self.attn1(l1)
            # att_map = torch.mean(c1, dim=2, keepdim=True)

        l2 = F.max_pool3d(l1, kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        l2 = self.ST_module2(l2)
        if "MyCBAM" in self.att_type:
            l2, c2 = self.attn2(l2)

        l3 = F.max_pool3d(l2, kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        l3 = self.ST_module4(l3)
        if "MyCBAM" in self.att_type:
            l3, c3 = self.attn3(l3)

        l4 = F.max_pool3d(l3, kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        g = self.spatialGlobalAvgpool(l4)

        #l5 = F.max_pool3d(l2, kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        #g2 = self.spatialGlobalAvgpool(l5)

        output = self.conv2d2(g)
        #output2 = self.conv2d2(g2)

        if "MyCBAM" not in self.att_type:
            c1, c2, c3 = None, None, None
        
        return [output, [l1,l2,l3]]
