import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Utils import plot_separated
from Params import Params
sys.path.pop()
import pdb
    
class Target_PR_Channel(object):
    
    def __init__(self, params:Params):
        self.PR_coefs = np.array(params.PR_coefs).reshape(1,-1)
        
        print('\nTarget Channel is')
        print(self.PR_coefs)
    
    def target_channel(self, codeword):
        target_channel_signal = (np.convolve(self.PR_coefs[0, :], codeword[0, :])
               [:-(self.PR_coefs.shape[1] - 1)].reshape(codeword.shape))
        
        return target_channel_signal
    
    def awgn(self, x, snr):
        E_b = 1
        sigma = np.sqrt(0.5 * E_b * 10 ** (- snr * 1.0 / 10))
        return x + sigma * np.random.normal(0, 1, x.shape)
    
if __name__ == '__main__':
    
    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()

    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    target_pr_channel = Target_PR_Channel(params)
    
    code_rate = 2/3
    Normalized_t = np.linspace(1, int(params.real_eval_len/code_rate), int(params.real_eval_len/code_rate))
        
    info = np.random.randint(2, size = (1, params.real_eval_len))
    codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
    pr_signal = target_pr_channel.target_channel(codeword)
    
    Xs = [
        Normalized_t,
        Normalized_t
    ]
    Ys = [
        {'data': codeword.reshape(-1), 'label': 'binary Sequence'}, 
        {'data': pr_signal.reshape(-1), 'label': 'pr_signal', 'color': 'red'},
    ]
    titles = [
        'Binary Sequence',
        'pr_signal',
    ]
    xlabels = ["Time (t/T)"]
    ylabels = [
        "Binary",
        "Amplitude",
    ]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )