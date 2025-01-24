import sys
import os
import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
sys.path.pop()

class RNN(nn.Module):
    def __init__(self, params:Params, device):
        super(RNN, self).__init__()
        self.params = params
        self.device = device
        self.time_step = (params.eval_length + params.overlap_length)
        self.fc_length = params.eval_length + params.overlap_length
        self.dec_input = torch.nn.Linear(params.input_size, 
                                         params.rnn_input_size)
        self.dec_rnn = torch.nn.GRU(params.rnn_input_size, 
                                    params.rnn_hidden_size, 
                                    params.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.rnn_dropout_ratio, 
                                    bidirectional=True)
        
        self.dec_output = torch.nn.Linear(2*params.rnn_hidden_size, params.output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        dec = torch.zeros(batch_size, self.fc_length, self.params.output_size).to(self.device)
        
        x = self.dec_input(x)
        y, _  = self.dec_rnn(x)
        y_dec = y[:, :self.time_step, :]

        dec = torch.sigmoid(self.dec_output(y_dec))
        
        return torch.squeeze(dec, 2)