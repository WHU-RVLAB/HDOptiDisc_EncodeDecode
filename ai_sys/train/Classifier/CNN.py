import sys
import os
import torch
import torch.nn as nn

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

class CNN(BaseModel):
    def __init__(self, params: Params, device):
        super(CNN, self).__init__(params, device)
        
        self.dec_input = nn.Linear(params.classifier_input_size, params.cnn_d_model)
        
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

        self.dec_output = nn.Linear(params.cnn_hidden_size, params.classifier_output_size)

    def forward(self, x):
        
        x_bt_size = x.shape[0]
        
        x = x.reshape(-1, self.params.classifier_input_size)
        
        x = self.dec_input(x)
        
        x = torch.unsqueeze(x, 2)
        
        x = self.dec_cnn(x)
        
        x = torch.squeeze(x, 2)
        
        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = x.reshape(x_bt_size, self.time_step, 1)
        
        x = torch.squeeze(x, 2)
    
        return x    