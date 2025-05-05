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
    
class Transformer(BaseModel):
    def __init__(self, params:Params, device):
        super(Transformer, self).__init__(params, device)
        self.dec_input = nn.Linear(params.classifier_input_size, params.transformer_d_model)
        transformer = nn.Transformer(
                                    d_model=params.transformer_d_model, 
                                    nhead=params.transformer_nhead, 
                                    dim_feedforward=params.transformer_hidden_size,
                                    num_encoder_layers=params.transformer_encoder_layers,
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.transformer_dropout_ratio)
                                    
        self.encoder = transformer.encoder
        self.dec_output = nn.Linear(params.transformer_d_model, params.classifier_output_size)
        
    def forward(self, x):
                
        x = self.dec_input(x)
        
        x = self.encoder(x)

        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = torch.squeeze(x, 2)
        
        return x