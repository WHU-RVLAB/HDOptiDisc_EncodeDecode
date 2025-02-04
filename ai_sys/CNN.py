import sys
import os
import torch
import torch.nn as nn
from BaseModel import BaseModel

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
sys.path.pop()

class CNN(BaseModel):
    def __init__(self, params: Params, device):
        super(CNN, self).__init__(params, device)
        
        self.dec_input = nn.Linear(params.input_size, params.cnn_d_model)
        
        self.dec_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=params.cnn_d_model,
                out_channels=params.cnn_hidden_size,
                kernel_size=params.cnn_kernel_size,
                padding='same'
            ),
            nn.ReLU(),
            nn.Dropout(params.cnn_dropout_ratio)
        )

        self.dec_output = nn.Linear(params.cnn_hidden_size, params.output_size)

    def forward(self, x):
        x = self.dec_input(x) 
        x = x.permute(0, 2, 1)
        y = self.dec_cnn(x)
        y = y.permute(0, 2, 1)
        dec = torch.sigmoid(self.dec_output(y))
        return torch.squeeze(dec, 2)