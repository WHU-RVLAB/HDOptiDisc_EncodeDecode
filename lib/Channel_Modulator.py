import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Utils import find_index
sys.path.pop()

## RLL_Modulator: constrained RLL(1,7) encoder
class RLL_Modulator(object):
    def __init__(self, encoder_dict, encoder_definite):
        self.encoder_dict = encoder_dict
        self.num_state = len(self.encoder_dict) # Num of states
        self.num_input_sym = self.encoder_dict[1]['input'].shape[1] # Num of input symbol
        self.num_out_sym = self.encoder_dict[1]['output'].shape[1] # Num of output symbol
        self.code_rate = self.num_input_sym / self.num_out_sym # Code rate
        self.ini_state = np.random.randint(low=1, high=self.num_state+1, size=1)[0] # Random initial state
        
    def forward_coding(self, info):
        '''
        Input: (1, length) array
        Output: (1, length / rate) array
        Mapping: Encoder (Markov Chain)
        '''
        
        info_len = np.size(info, 1)
        codeword = np.zeros((1, int(info_len/self.code_rate)))
        
        state = self.ini_state
        for i in range(0, info_len, self.num_input_sym):
            # start symbol and state
            idx = int(i / self.num_input_sym)
            input_sym = info[:, i:i+self.num_input_sym][0]
            # input idx
            idx_in = find_index(self.encoder_dict[state]['input'], input_sym)
            # output sym and next state
            output_sym = self.encoder_dict[state]['output'][idx_in, :]
            state = self.encoder_dict[state]['next_state'][idx_in, 0]
            codeword[:, self.num_out_sym*idx : self.num_out_sym*(idx+1)] = output_sym
        
        return codeword.astype(int)
    
    def inverse_coding(self, info):
        pass