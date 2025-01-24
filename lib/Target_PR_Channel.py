import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine, Target_channel_state_machine, Target_channel_dummy_bits
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Utils import plot_separated
from Params import Params
sys.path.pop()
import pdb
    
class Target_PR_Channel(object):
    
    def __init__(self, channel_dict, dummy_list, ini_state):
        self.channel_dict = channel_dict
        self.dummy_list = dummy_list
        self.ini_state = ini_state
        
        self.len_dummy = self.dummy_list[0].shape[1]
        self.num_input_sym = int(self.channel_dict['in_out'].shape[1] / 2)
    
    def target_channel(self, x):
        '''
        Input: (1, length) array
        Output: (1, len + dummy_len) array
        Mapping: channel state machine to zero state
        '''
        
        # remember 5 dummy values in the end
        length = x.shape[1] - self.len_dummy
        y = np.zeros((1, x.shape[1]))
        
        # Memory channel
        state = self.ini_state
        for i in range(0, length, self.num_input_sym):
            set_in = np.where(self.channel_dict['state_machine'][:, 0]==state)[0]
            idx_in = set_in[np.where(self.channel_dict['in_out'][set_in, 0]==x[:, i])[0]]
            y[:, i] = self.channel_dict['in_out'][idx_in, 1]
            state = self.channel_dict['state_machine'][idx_in, 1]
        
        # Dummy bits to zero state
        path_dummy = self.dummy_list[state[0]]
        for i in range(0, self.len_dummy, self.num_input_sym):
            set_in = np.where(self.channel_dict['state_machine'][:, 0]==state)[0]
            idx_in = (set_in[np.where(self.channel_dict['state_machine'][set_in, 1]==
                                      path_dummy[:, i])[0]])
            y[:, i+length] = self.channel_dict['in_out'][idx_in, 1]
            state = path_dummy[:, i]
        
        return y
    
    def awgn(self, x, snr):
        E_b = 1
        sigma = np.sqrt(0.5 * E_b * 10 ** (- snr * 1.0 / 10))
        return x + sigma * np.random.normal(0, 1, x.shape)
    
if __name__ == '__main__':
    
    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()
    dummy_start_paths, dummy_start_input, dummy_start_output, dummy_start_eval, \
    dummy_end_paths, dummy_end_input, dummy_end_output, dummy_end_eval = Target_channel_dummy_bits()

    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    target_pr_channel = Target_PR_Channel(channel_dict, dummy_end_paths, channel_dict['ini_state'])
    
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