import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine
from Channel_Modulator import RLL_Modulator
from Params import Params
sys.path.pop()

class NRZI_Converter(object):
    
    def __init__(self):
        pass
    
    def forward_coding(self, z):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: x = (1 / 1 + D) z (mod 2)
        x_{-1} = 0
        '''
        
        length = np.size(z, 1)
        x = np.zeros((1, length))
        x[0, 0] = z[0, 0]
        for i in range(1, length):
            x[0, i] = x[0, i-1] + z[0, i]
        return x % 2
    
    def inverse_coding(self, x):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: x = (1 + D) z (mod 2)
        z_{-1} = 0
        '''
    
        length = x.shape[1]
        z = np.zeros((1, length))
        z[0, 0] = x[0, 0]
        for i in range(1, length):
            z[0, i] = x[0, i] + x[0, i-1]
        return z % 2
    
if __name__ == '__main__':
    
    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
        
    info = np.random.randint(2, size = (1, params.data_val_len))
    RLL_codeword = RLL_modulator.forward_coding(info)
    NRZI_codeword = NRZI_converter.forward_coding(RLL_codeword)
    
    print("\ninfo: ", info)
    print("\ninfo.shape: ", info.shape)
    print("\nRLL_codeword: ", RLL_codeword)
    print("\nRLL_codeword.shape: ", RLL_codeword.shape)
    print("\nNRZI_codeword: ", NRZI_codeword)
    print("\nNRZI_codeword.shape: ", NRZI_codeword.shape)