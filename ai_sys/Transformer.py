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
    
class Transformer(BaseModel):
    def __init__(self, params:Params, device):
        super(Transformer, self).__init__(params, device)
        self.dec_input = nn.Linear(params.input_size, params.transformer_d_model)
        transformer = nn.Transformer(
                                    d_model=params.transformer_d_model, 
                                    nhead=params.transformer_nhead, 
                                    dim_feedforward=params.transformer_hidden_size,
                                    num_encoder_layers=params.transformer_encoder_layers,
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.transformer_dropout_ratio)
                                    
        self.encoder = transformer.encoder
        self.dec_output = nn.Linear(params.transformer_d_model, params.output_size)
        
    def forward(self, x):        
        x = self.dec_input(x)
        y = self.encoder(x)
        y_dec = y[:, :self.time_step, :]

        dec = torch.sigmoid(self.dec_output(y_dec))
        
        return torch.squeeze(dec, 2)