import sys
import os
import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import codeword_threshold
sys.path.pop()

class RNN(nn.Module):
    def __init__(self, params:Params, device):
        super(RNN, self).__init__()
        self.params = params
        self.device = device
        self.time_step = params.eval_length + params.overlap_length
        self.fc_length = params.eval_length + params.overlap_length
        self.dec_input = torch.nn.Linear(params.inputx_size, params.rnn_d_model)
        self.dec_rnn = torch.nn.GRU(params.rnn_d_model, 
                                    params.rnn_hidden_size, 
                                    params.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.rnn_dropout_ratio, 
                                    bidirectional=True)
        
        self.dec_output = torch.nn.Linear(2*params.rnn_hidden_size, params.output_size)
        
    def forward(self, x):        
        x = self.dec_input(x)
        y, _  = self.dec_rnn(x)
        y_dec = y[:, :self.time_step, :]

        dec = torch.sigmoid(self.dec_output(y_dec))
        
        return torch.squeeze(dec, 2)
    
    def decode(self, eval_length, data_eval, device):
        data_eval = data_eval.to(device)
        dec = torch.zeros((1, 0)).float().to(device)
        for idx in range(data_eval.shape[0]):
            truncation_in = data_eval[idx:idx + 1, : , :]
            with torch.no_grad():
                dec_block = codeword_threshold(self.forward(truncation_in)[:, :eval_length])
            # concatenate the decoding codeword
            dec = torch.cat((dec, dec_block), 1)
            
        return dec.cpu().numpy()