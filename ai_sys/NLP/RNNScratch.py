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

class RNNScratch(BaseModel):
    def __init__(self, params: Params, device):
        super(RNNScratch, self).__init__(params, device)
        
        self.input_dim = params.nlp_input_size
        self.hidden_dim = params.rnn_hidden_size
        self.output_dim = params.nlp_output_size

        self.W_xh = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim, requires_grad=True, device=device) * 0.01)
        
        self.W_hh = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, requires_grad=True, device=device) * 0.01)
        
        self.b_h = nn.Parameter(torch.zeros(self.hidden_dim, requires_grad=True, device=device))

        self.W_hy = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim, requires_grad=True, device=device) * 0.01)
        
        self.b_y = nn.Parameter(torch.zeros(self.output_dim, requires_grad=True, device=device))

    def forward(self, x, h):
        batch_size, seq_length, _ = x.shape
        hidden_states = []
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            h = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)
            hidden_states.append(h)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        
        y = hidden_states @ self.W_hy + self.b_y
        y = torch.sigmoid(y)
        y = torch.squeeze(y, 2)

        return y, hidden_states