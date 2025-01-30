import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import copy

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import subsequent_mask
sys.path.pop()

# Embeddings and Softmax
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embedds = self.lut(x)
        return embedds * torch.tensor(math.sqrt(self.d_model), device=embedds.device)

# Positional Encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, sentence_len, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(sentence_len, d_model)
        position = torch.arange(0, sentence_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe = pe.unsqueeze(0)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1), :], requires_grad=False)
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
        self.src_embed = nn.Sequential(Embeddings(params.transformer_d_model, params.transformer_src_vocab), c(position))    # input embedding module(input embedding + positional encode)
        self.tgt_embed = nn.Sequential(Embeddings(params.transformer_d_model, params.transformer_tgt_vocab), c(position))    # ouput embedding module
        self.src_position = c(position)
        self.generator = Generator(params.transformer_d_model, params.transformer_tgt_vocab)   # output generation module
        
    def forward(self, src, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src)
        res = self.decode(memory, tgt, tgt_mask)
        return res

    def encode(self, src):
        # feature extract
        src_embedds = self.src_embed(src)

        # position encode
        src_embedds = self.src_position(src_embedds)

        return self.encoder(src_embedds)

    def decode(self, memory, tgt, tgt_mask):
        target_embedds = self.tgt_embed(tgt)
        return self.decoder(memory, target_embedds, tgt_mask)

    def greedy_decode(self, src, max_len, start_symbol, end_symbol):
        memory = self.encode(src)
        # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
        for i in range(max_len-1):
            out = self.decode(memory,
                            Variable(ys),
                            Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            next_word = torch.ones(1, 1).type_as(src.data).fill_(next_word).long()
            ys = torch.cat([ys, next_word], dim=1)

            next_word = int(next_word)
            if next_word == end_symbol:
                break
            #ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        ys = ys[0, 1:]
        return ys.cpu()
