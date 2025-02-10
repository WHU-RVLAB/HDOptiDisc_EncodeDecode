import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from BaseModel import BaseModel
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
sys.path.pop()

class UNet1D(BaseModel):
    def __init__(self, params:Params, device):
        super(UNet1D, self).__init__(params, device)
        
        self.enc1 = self.conv_block(params.unet_d_model, params.unet_base_filters)
        self.enc2 = self.conv_block(params.unet_base_filters, params.unet_base_filters * 2)
        self.enc3 = self.conv_block(params.unet_base_filters * 2, params.unet_base_filters * 4)
        self.enc4 = self.conv_block(params.unet_base_filters * 4, params.unet_base_filters * 8)
        
        self.bottleneck = self.conv_block(params.unet_base_filters * 8, params.unet_base_filters * 16)
        
        self.up4 = nn.ConvTranspose1d(params.unet_base_filters * 16, params.unet_base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(params.unet_base_filters * 16, params.unet_base_filters * 8)
        
        self.up3 = nn.ConvTranspose1d(params.unet_base_filters * 8, params.unet_base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(params.unet_base_filters * 8, params.unet_base_filters * 4)
        
        self.up2 = nn.ConvTranspose1d(params.unet_base_filters * 4, params.unet_base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(params.unet_base_filters * 4, params.unet_base_filters * 2)
        
        self.up1 = nn.ConvTranspose1d(params.unet_base_filters * 2, params.unet_base_filters, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(params.unet_base_filters * 2, params.unet_base_filters)
        
        self.out_conv = nn.Conv1d(params.unet_base_filters, params.output_size, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool1d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool1d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool1d(e3, kernel_size=2))
        
        b = self.bottleneck(F.max_pool1d(e4, kernel_size=2))
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
    
        d1 = d1[:, :self.time_step, :]

        out = torch.sigmoid(self.out_conv(d1))
        
        return torch.squeeze(out, 2)
