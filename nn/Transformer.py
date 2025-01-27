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

class Transformer(nn.Module):

    def __init__(self, params:Params, device):
        super(Transformer, self).__init__()
        self.params = params
        self.device = device
        self.time_step = params.eval_length + params.overlap_length
        self.fc_length = params.eval_length + params.overlap_length
        self.dec_input = torch.nn.Linear(params.input_size, 
                                         params.transformer_input_size)
        self.dec_transformer = torch.nn.Transformer(
                                    d_model=params.transformer_input_size, 
                                    nhead=params.transformer_nhead, 
                                    dim_feedforward=params.transformer_hidden_size,
                                    num_encoder_layers=params.transformer_encoder_layers,
                                    # num_decoder_layers=params.transformer_decoder_layers,
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.transformer_dropout_ratio)
        
        self.dec_output = torch.nn.Linear(params.transformer_input_size, params.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        dec = torch.zeros(batch_size, self.fc_length, self.params.output_size).to(self.device)
        
        x = self.dec_input(x)
        y = self.dec_transformer.encoder(x, mask=None)
        y_dec = y[:, :self.time_step, :]

        dec = torch.sigmoid(self.dec_output(y_dec))
        
        return torch.squeeze(dec, 2)
