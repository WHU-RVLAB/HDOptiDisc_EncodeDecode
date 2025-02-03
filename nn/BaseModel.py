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

class BaseModel(nn.Module):
    def __init__(self, params:Params, device):
        super(BaseModel, self).__init__()
        self.params = params
        self.device = device
        self.time_step = params.eval_length + params.overlap_length
        self.fc_length = params.eval_length + params.overlap_length
        
    def forward(self, x):
        pass
    
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