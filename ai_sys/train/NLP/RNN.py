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

class RNN(BaseModel):
    def __init__(self, params:Params, device):
        super(RNN, self).__init__(params, device)
        self.hidden_size = params.rnn_hidden_size
        self.batch_first = True
        self.bidirectional = params.rnn_bidirectional
        self.num_directions = 2 if params.rnn_bidirectional else 1
        
        self.dec_rnn_forward = nn.RNNCell(
            params.rnn_d_model, 
            params.rnn_hidden_size, 
            nonlinearity='relu',
            bias=True)
        self.dec_rnn_inverse = nn.RNNCell(
            params.rnn_d_model, 
            params.rnn_hidden_size, 
            nonlinearity='relu',
            bias=True)
        self.dec_rnn = [self.dec_rnn_forward, self.dec_rnn_inverse]
        
        rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
        self.dec_output = nn.Linear(rnn_hidden_size_factor*params.rnn_hidden_size, params.nlp_output_size)
        
    def rnn_forward(self, x, h_0=None):

        if self.batch_first:
            x = x.transpose(0, 1)  # [T, B, input_size]
    
        seq_len, batch_size, _ = x.shape

        if h_0 is None:
            h_0 = torch.zeros((self.num_directions, batch_size, self.hidden_size)).to(x.device)

        outputs = []
        h_n = torch.zeros_like(h_0).to(x.device)

        for direction in range(self.num_directions):
            h_t = h_0[direction]
            input_seq = x if direction == 0 else torch.flip(x, dims=[0])
            step_outputs = []

            for t in range(seq_len):
                step_input = input_seq[t]
                h_t = self.dec_rnn[direction](step_input, h_t)
                step_outputs.append(h_t)

            if direction == 1:
                step_outputs = step_outputs[::-1]
        
            outputs.append(torch.stack(step_outputs, axis=0))  # [T, B, H]
            h_n[direction] = h_t

        if self.bidirectional:
            output = torch.concatenate(outputs, axis=2)  # [T, B, 2H]
        else:
            output = outputs[0]  # [T, B, H]

        if self.batch_first:
            output = output.transpose(0, 1)  # [B, T, H or 2H]

        return output, h_n
       
    def forward(self, x, h): 
                
        x, h  = self.rnn_forward(x, h)
        
        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = torch.squeeze(x, 2)
        
        return x, h