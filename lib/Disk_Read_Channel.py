import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Disk_Response import BD_symbol_response
from Utils import plot_separated
from Params import Params
sys.path.pop()
import pdb
    
class Disk_Read_Channel(object):
    
    def __init__(self, params:Params):
        self.params = params
        _, bd_di_coef = BD_symbol_response(bit_periods = 10)
        mid_idx = len(bd_di_coef)//2
        self.bd_di_coef = bd_di_coef[mid_idx : mid_idx + self.params.tap_bd_num].reshape(1,-1)
        
        print('\nThe dipulse bd coefficient is')
        print(bd_di_coef)
        print('\nTap bd coefficient is')
        print(self.bd_di_coef)
    
    def RF_signal(self, codeword):
        rf_signal = (np.convolve(self.bd_di_coef[0, :], codeword[0, :])
               [:-(self.params.tap_bd_num - 1)].reshape(codeword.shape))
        
        return rf_signal
    
    def awgn(self, x, snr):
        E_b = np.mean(np.square(x[0, :self.params.truncation4energy]))
        sigma = np.sqrt(0.5 * E_b * 10 ** (- snr * 1.0 / 10))
        return x + sigma * np.random.normal(0, 1, x.shape)
    
    def jitter(self, x, zeta):
        x_padded = np.pad(x, ((0, 0), (0, 1)), 'constant', constant_values=0)
        x_d = np.diff(x_padded, axis=1)
        
        x_d_padded = np.pad(x_d, ((0, 0), (0, 1)), 'constant', constant_values=0)
        x_d2 = np.diff(x_d_padded, axis=1)
        
        return x + zeta*x_d + 0.5*pow(zeta,2)*x_d2
    
if __name__ == '__main__':
    
    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel(params)
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    codeword_len = int(params.equalizer_train_len/rate_constrain)
    
    params.snr_step = (params.snr_stop-params.snr_start)/(params.num_plots - 1)
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step + 1)
    
    Normalized_t = np.linspace(1, int(params.data_val_len/rate_constrain), int(params.data_val_len/rate_constrain))
    
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.data_val_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        rf_signal = disk_read_channel.RF_signal(codeword)
        equalizer_input = disk_read_channel.awgn(rf_signal, snr)
        equalizer_input_jitter = disk_read_channel.jitter(equalizer_input, params.zeta)
        
        Xs = [
            Normalized_t,
            Normalized_t,
            Normalized_t,
            Normalized_t
        ]
        Ys = [
           {'data': codeword.reshape(-1), 'label': 'binary Sequence'}, 
           {'data': rf_signal.reshape(-1), 'label': 'rf_signal', 'color': 'red'},
           {'data': equalizer_input.reshape(-1), 'label': f'equalizer_input_snr{snr}', 'color': 'red'},
           {'data': equalizer_input_jitter.reshape(-1), 'label': f'equalizer_input_jitter_zeta{params.zeta}', 'color': 'red'}
        ]
        titles = [
            'Binary Sequence',
            'rf_signal',
            f'equalizer_input_snr{snr}',
            f'equalizer_input_jitter_zeta{params.zeta}',
        ]
        xlabels = ["Time (t/T)"]
        ylabels = [
            "Binary",
            "Amplitude",
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