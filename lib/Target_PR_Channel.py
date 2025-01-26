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
        self.params = params
        self.PR_coefs = np.array(params.PR_coefs).reshape(1,-1)
        
        print('\nTarget Channel is')
        print(self.PR_coefs)
    
    def target_channel(self, codeword):
        target_channel_signal = (np.convolve(self.PR_coefs[0, :], codeword[0, :])
               [:-(self.PR_coefs.shape[1] - 1)].reshape(codeword.shape))
        
        return target_channel_signal
    
    def awgn(self, x, snr):
        E_b = np.mean(np.square(x[0, :self.params.truncation4energy]))
        sigma = np.sqrt(0.5 * E_b * 10 ** (- snr * 1.0 / 10))
        return x + sigma * np.random.normal(0, 1, x.shape)
    
if __name__ == '__main__':
    
    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()

    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    codeword_len = int(params.equalizer_train_len/rate_constrain)
    
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    target_pr_channel = Target_PR_Channel(params)
    
    params.snr_step = (params.snr_stop-params.snr_start)/(params.num_plots - 1)
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step + 1)
    
    Normalized_t = np.linspace(1, int(params.real_eval_len/rate_constrain), int(params.real_eval_len/rate_constrain))
    
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.real_eval_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        pr_signal = target_pr_channel.target_channel(codeword)
        pr_signal_noise = target_pr_channel.awgn(pr_signal, snr)
        
        Xs = [
            Normalized_t,
            Normalized_t,
            Normalized_t
        ]
        Ys = [
            {'data': codeword.reshape(-1), 'label': 'binary Sequence'}, 
            {'data': pr_signal.reshape(-1), 'label': 'pr_signal', 'color': 'red'},
            {'data': pr_signal_noise.reshape(-1), 'label': f'pr_signal_noise{snr}', 'color': 'red'}
        ]
        titles = [
            'Binary Sequence',
            'pr_signal',
            f'pr_signal_noise{snr}'
        ]
        xlabels = ["Time (t/T)"]
        ylabels = [
            "Binary",
            "Amplitude",
            "Amplitude"
        ]
        plot_separated(
            Xs=Xs, 
            Ys=Ys, 
            titles=titles,     
            xlabels=xlabels, 
            ylabels=ylabels
        )