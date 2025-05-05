import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from BaseModel import BaseModel
sys.path.pop()

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))))
from lib.Params import Params
sys.path.pop()

class Conv1d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(Conv1d_bn,self).__init__()

        self.conv1 = nn.Conv1d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv1d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class Deconv1d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(Deconv1d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
    
class UNet1D(BaseModel):
    def __init__(self, params:Params, device):
        
        super(UNet1D, self).__init__(params, device)
        
        self.dec_input = nn.Linear(params.classifier_input_size, params.unet_d_model)
        
        # encoder
        self.layer1_conv = Conv1d_bn(1, params.unet_base_filters)
        self.layer2_conv = Conv1d_bn(params.unet_base_filters, params.unet_base_filters * 2)
        self.layer3_conv = Conv1d_bn(params.unet_base_filters * 2, params.unet_base_filters * 4)
        self.layer4_conv = Conv1d_bn(params.unet_base_filters * 4, params.unet_base_filters * 8)
        
        # bottleneck
        self.layer5_conv = Conv1d_bn(params.unet_base_filters * 8, params.unet_base_filters * 16)
        
        # decoder
        self.layer6_conv = Conv1d_bn(params.unet_base_filters * 16, params.unet_base_filters * 8)
        self.layer7_conv = Conv1d_bn(params.unet_base_filters * 8, params.unet_base_filters * 4)
        self.layer8_conv = Conv1d_bn(params.unet_base_filters * 4, params.unet_base_filters * 2)
        self.layer9_conv = Conv1d_bn(params.unet_base_filters * 2, params.unet_base_filters)
        self.layer10_conv = nn.Conv1d(params.unet_base_filters,1,kernel_size=3, stride=1,padding=1,bias=True)
        
        # decoder
        self.deconv1 = Deconv1d_bn(params.unet_base_filters * 16, params.unet_base_filters * 8)
        self.deconv2 = Deconv1d_bn(params.unet_base_filters * 8, params.unet_base_filters * 4)
        self.deconv3 = Deconv1d_bn(params.unet_base_filters * 4, params.unet_base_filters * 2)
        self.deconv4 = Deconv1d_bn(params.unet_base_filters * 2, params.unet_base_filters)
        
        self.dec_output = nn.Linear(params.unet_d_model, params.classifier_output_size)
        
    def forward(self,x):
        
        x_bt_size = x.shape[0]
        
        x = x.reshape(-1, self.params.classifier_input_size)
        
        x = self.dec_input(x)
        
        x = torch.unsqueeze(x, 1)
        
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool1d(conv1,2)
        
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool1d(conv2,2)
        
        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool1d(conv3,2)
        
        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool1d(conv4,2)
        
        conv5 = self.layer5_conv(pool4)
        
        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.layer6_conv(concat1)
        
        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,conv1],dim=1)
        conv9 = self.layer9_conv(concat4)
        
        x = self.layer10_conv(conv9)
        
        x = torch.squeeze(x, 1)
        
        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = x.reshape(x_bt_size, self.time_step, 1)
        
        x = torch.squeeze(x, 2)
    
        return x