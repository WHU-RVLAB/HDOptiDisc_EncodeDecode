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

class RNN(BaseModel):
    def __init__(self, params:Params, device):
        super(RNN, self).__init__(params, device)
        self.dec_input = nn.Linear(params.input_size, params.rnn_d_model)
        self.dec_rnn = nn.GRU(params.rnn_d_model, 
                                    params.rnn_hidden_size, 
                                    params.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.rnn_dropout_ratio, 
                                    bidirectional=True)
        
        self.dec_output = nn.Linear(2*params.rnn_hidden_size, params.output_size)
        
    def forward(self, x): 
        
        x = self.dec_input(x)
        
        x, _  = self.dec_rnn(x)

        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = torch.squeeze(x, 2)
        
        return x