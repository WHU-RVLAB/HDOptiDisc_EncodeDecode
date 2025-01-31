import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import subsequent_mask
sys.path.pop()

# Positional Encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, sentence_len, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(1, sentence_len, d_model)
        position = torch.arange(0, sentence_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) / d_model) * (-math.log(10000.0)))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.pe = pe
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)
        
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
# Model Architecture
class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    Base for this and many other models.
    """
    def __init__(self, params:Params, device):
        super(Transformer, self).__init__()
        self.params = params
        self.device = device
        self.time_step = params.eval_length + params.overlap_length
        self.fc_length = params.eval_length + params.overlap_length
        c = copy.deepcopy
        position = PositionalEncoding(params.transformer_decode_max_len, params.transformer_d_model, params.transformer_dropout_ratio)
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
        self.src_embed = nn.Sequential(nn.Embedding(params.transformer_src_vocab, params.transformer_d_model), c(position))    # input embedding module(input embedding + positional encode)
        self.tgt_embed = nn.Sequential(nn.Embedding(params.transformer_tgt_vocab, params.transformer_d_model), c(position))    # ouput embedding module
        self.generator = Generator(params.transformer_d_model, params.transformer_tgt_vocab)   # output generation module
        
    def forward(self, src, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src)
        res = self.decode(memory, tgt, tgt_mask)
        return res

    def encode(self, src):
        src_embedds = self.src_embed(src)
        return self.encoder(src_embedds)

    def decode(self, memory, tgt, tgt_mask):
        target_embedds = self.tgt_embed(tgt)
        return self.decoder(target_embedds, memory, tgt_mask)

    def greedy_decode(self, src, target_pred, max_len, start_symbol, end_symbol):
        memory = self.encode(src)
        target_input_origin = torch.ones(src.shape[0], 1).fill_(start_symbol).type_as(src.data).to(src.device)
        target_input = target_input_origin
        
        for i in range(max_len-1):
            target_mask = torch.from_numpy(subsequent_mask(src.shape[0], target_input.size(1))).to(src.device)
            target_mask = target_mask.repeat(self.params.transformer_nhead, 1, 1)
            
            out = self.decode(memory, target_input, target_mask)
            prob = self.generator(out)
            _, next_word = torch.max(prob, dim = 2)
            
            target_input = torch.cat([target_input_origin, next_word[:, :]], dim=1)

        target_input = target_input[:, 1:]
        return target_input.cpu()
