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

class MLP(BaseModel):
    def __init__(self, params:Params, device):
        super(MLP, self).__init__(params, device)
        
        self.dec_input = nn.Sequential(nn.Linear(params.input_size, params.mlp_d_model), 
                                       nn.ReLU(),
                                       nn.Dropout(params.mlp_dropout_ratio))
        self.dec_mlp = nn.Sequential(nn.Linear(params.mlp_d_model, params.mlp_hidden_size), 
                                     nn.ReLU(),
                                     nn.Dropout(params.mlp_dropout_ratio))
        self.dec_output = nn.Linear(params.mlp_hidden_size, params.output_size)
        
    def forward(self, x): 
        
        x_bt_size = x.shape[0]
        
        x = x.reshape(-1, self.params.input_size)
               
        x = self.dec_input(x)
        
        x = self.dec_mlp(x)
        
        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = x.reshape(x_bt_size, self.time_step, 1)
        
        x = torch.squeeze(x, 2)
        
        return x