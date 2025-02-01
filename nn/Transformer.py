import sys
import os
import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import convert2transformer, codeword_threshold
sys.path.pop()
    
class Transformer(nn.Module):
    def __init__(self, params:Params, device):
        super(Transformer, self).__init__()
        self.params = params
        self.device = device
        self.time_step = params.eval_length + params.overlap_length
        self.fc_length = params.eval_length + params.overlap_length
        self.decx_input = torch.nn.Linear(params.inputx_size, params.transformer_d_model)
        self.decy_input = torch.nn.Linear(params.inputy_size, params.transformer_d_model)
        transformer = torch.nn.Transformer(
                                    d_model=params.transformer_d_model, 
                                    nhead=params.transformer_nhead, 
                                    dim_feedforward=params.transformer_hidden_size,
                                    num_encoder_layers=params.transformer_encoder_layers,
                                    num_decoder_layers=params.transformer_decoder_layers,
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.transformer_dropout_ratio)
                                    
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder
        self.dec_output = torch.nn.Linear(params.transformer_d_model, params.output_size)
        
    def forward(self, x, y, y_mask):
        x = self.decx_input(x)
        memory = self.encoder(x)
        
        y = self.decy_input(y)
        y = self.decoder(y, memory, tgt_mask = y_mask)

        dec = torch.sigmoid(self.dec_output(y))
        
        return torch.squeeze(dec, 2)

    def decode(self, eval_length, data_eval, device):
        dec = torch.zeros((1, 0)).float().to(device)
        for idx in range(data_eval.shape[0]):
            x = data_eval[idx:idx + 1, : , :]
            
            y_origin = torch.zeros((x.shape[0], 0)).float().to(device)
            y = y_origin
            
            for _ in range(eval_length):
                src, target_input, target_pred, target_mask = \
                    convert2transformer(x, y, self.params.transformer_nhead, start=2, device=device)
                
                with torch.no_grad():
                    y_pred = codeword_threshold(self.forward(src, target_input, target_mask)[:, :eval_length])
                
                y = torch.cat([y, y_pred[:, -1:]], dim=1)
            
            # concatenate the decoding codeword
            dec_block = y
            dec = torch.cat((dec, dec_block), 1)
            
        return dec.cpu().numpy()