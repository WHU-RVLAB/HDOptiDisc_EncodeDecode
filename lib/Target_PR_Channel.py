import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine, Target_channel_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Utils import plot_separated
sys.path.pop()
import pdb

info_len = 1000
    
class Target_PR_Channel(object):
    
    def __init__(self, channel_machine, dummy_list, ini_state):
        self.channel_machine = channel_machine
        self.dummy_list = dummy_list
        self.ini_state = ini_state
        
        self.len_dummy = self.dummy_list[0].shape[1]
        self.num_input_sym = int(self.channel_machine['in_out'].shape[1] / 2)
        
    # def __init__(self, target_pr_coefs):
    #     _, target_channel_coef = partial_response(PR_coefs = target_pr_coefs)
    #     tap_pr_num = len(target_pr_coefs)
    #     target_channel_coef = target_channel_coef[len(target_channel_coef)//2:len(target_channel_coef)//2 + tap_pr_num]
    #     self.target_channel_coef = target_channel_coef.reshape(1,-1)
    #     self.tap_pr_num = tap_pr_num
    #     print('The target pr channel coefficient is\n')
    #     print(self.target_channel_coef)
    
    # def PR_signal(self, codeword):
    #     tap_pr_num_side = int((self.tap_pr_num - 1) / 2)
    #     pr_signal = (np.convolve(self.target_channel_coef[0, :], codeword[0, :])
    #            [tap_pr_num_side:-tap_pr_num_side].reshape(codeword.shape))
        
    #     return pr_signal
    
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
            set_in = np.where(self.channel_machine['state_machine'][:, 0]==state)[0]
            idx_in = set_in[np.where(self.channel_machine['in_out'][set_in, 0]==x[:, i])[0]]
            y[:, i] = self.channel_machine['in_out'][idx_in, 1]
            state = self.channel_machine['state_machine'][idx_in, 1]
        
        # Dummy bits to zero state
        path_dummy = self.dummy_list[state[0]]
        for i in range(0, self.len_dummy, self.num_input_sym):
            set_in = np.where(self.channel_machine['state_machine'][:, 0]==state)[0]
            idx_in = (set_in[np.where(self.channel_machine['state_machine'][set_in, 1]==
                                      path_dummy[:, i])[0]])
            y[:, i+length] = self.channel_machine['in_out'][idx_in, 1]
            state = path_dummy[:, i]
        
        return y
    
    def awgn(self, x, snr):
        E_b = 1
        sigma = np.sqrt(0.5 * E_b * 10 ** (- snr * 1.0 / 10))
        return x + sigma * np.random.normal(0, 1, x.shape)
    
if __name__ == '__main__':
    
    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict, dummy_dict, ini_metric = Target_channel_state_machine()
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    target_pr_channel = Target_PR_Channel(channel_dict, dummy_dict, channel_dict['ini_state'])
    
    code_rate = 2/3
    Normalized_t = np.linspace(1, int(info_len/code_rate), int(info_len/code_rate))
        
    info = np.random.randint(2, size = (1, info_len))
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