import sys
import os
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))))
from lib.Params import Params
from lib.Utils import codeword_threshold
sys.path.pop()

class BaseModel(object):
    def __init__(self, params:Params):
        super(BaseModel, self).__init__()
        self.params = params
        self.time_step = params.pre_overlap_length + params.eval_length + params.post_overlap_length
        
    def forward(self, x):
        pass
    
    def decode(self, data_eval, hidden_state):
        dec = np.zeros((data_eval.shape[0], 0))
        
        decodeword, hidden_state = self.forward(data_eval, hidden_state)
        dec_block = codeword_threshold(decodeword)
        # concatenate the decoding codeword
        dec = np.concatenate((dec, dec_block), axis=1)
            
        return dec.reshape(1, -1), hidden_state