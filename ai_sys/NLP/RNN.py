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
                os.path.abspath(__file__)))))
from lib.Params import Params
sys.path.pop()

class RNN(BaseModel):
    def __init__(self, params:Params, device):
        super(RNN, self).__init__(params, device)
        self.dec_input = nn.Linear(params.nlp_input_size, params.rnn_d_model)
        self.dec_rnn = nn.GRU(params.rnn_d_model, 
                                    params.rnn_hidden_size, 
                                    params.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.rnn_dropout_ratio, 
                                    bidirectional=params.rnn_bidirectional)
        
        rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
        self.dec_output = nn.Linear(rnn_hidden_size_factor*params.rnn_hidden_size, params.nlp_output_size)
        
    def forward(self, x, h): 
        
        x = self.dec_input(x)
        
        x, h  = self.dec_rnn(x, h)

        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = torch.squeeze(x, 2)
        
        return x, h