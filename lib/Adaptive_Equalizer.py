import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine, Target_channel_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Disk_Read_Channel import Disk_Read_Channel
from Target_PR_Channel import Target_PR_Channel
from Utils import plot_separated
sys.path.pop()

import pdb

info_len = 100
snr = 40
BD_bits_freq = 132e6

class Adaptive_Equalizer(object):
    
    def __init__(self, equalizer_input, reference_signal, taps_num, mu):
        self.equalizer_input = equalizer_input
        self.reference_signal = reference_signal
        self.taps_num = taps_num
        self.equalizer_coeffs = np.zeros(self.taps_num)
        self.mu = mu
        
    def lms(self):
        
        # number of iterations for adaptive equalization
        N = len(self.equalizer_input) - self.taps_num + 1
        
        # get the delay for the FIR filter
        delay = (self.taps_num-1) // 2

        equalizer_output = np.zeros(len(self.equalizer_input))
        error_signal = np.zeros(len(self.equalizer_input))
        error_signal_square = np.zeros(len(self.equalizer_input))
        
        # perform the adaptive equalizering
        for n in range(N):
            
            # sliding window corresponding to equalizer order
            x = np.flipud(self.equalizer_input[n:n+self.taps_num])

            # calculate the output at time n
            equalizer_output[n] = np.dot(x, self.equalizer_coeffs)
            
            # calculate the error
            error_signal[n] = self.reference_signal[n + delay] - equalizer_output[n]
            
            # calculate the new equalizer coefficients
            self.equalizer_coeffs += 2 * self.mu * error_signal[n] * x 

            # calculate the squared error
            error_signal_square[n] = error_signal[n]**2

        return equalizer_output, error_signal, error_signal_square, self.equalizer_coeffs
    
    def equalized_signal(self):
        
        # number of iterations for adaptive equalization
        N = len(self.equalizer_input) - self.taps_num + 1
        equalizer_output = np.zeros(len(self.equalizer_input))
        
        for n in range(N):
            x = np.flipud(self.equalizer_input[n:n+self.taps_num])
            equalizer_output[n] = np.dot(x, self.equalizer_coeffs)
            
        return equalizer_output

if __name__ == '__main__':
    
    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict, dummy_dict, ini_metric = Target_channel_state_machine()
    
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel()
    target_pr_channel = Target_PR_Channel(channel_dict, dummy_dict, channel_dict['ini_state'])
    
    code_rate = 2/3
    Normalized_t = np.linspace(1, int(info_len/code_rate), int(info_len/code_rate))
        
    info = np.random.randint(2, size = (1, info_len))
    codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
    
    rf_signal = disk_read_channel.RF_signal(codeword)
    equalizer_input = disk_read_channel.awgn(rf_signal, snr)
    
    pr_signal = target_pr_channel.target_channel(codeword)
    
    pr_adaptive_equalizer = Adaptive_Equalizer(        
        equalizer_input  = equalizer_input.reshape(-1),
        reference_signal = pr_signal.reshape(-1),
        taps_num = 9,
        mu = 0.01
    )
    detector_input, error_signal, error_signal_square, equalizer_coeffs = pr_adaptive_equalizer.lms()
    
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
        {'data': pr_signal.reshape(-1), 'label': 'pr_signal', 'color': 'red'},
    ]
    titles = [
        'Binary Sequence',
        'rf_signal',
        f'equalizer_input_snr{snr}',
        'pr_signal',
    ]
    xlabels = ["Time (t/T)"]
    ylabels = [
        "Binary",
        "Amplitude",
        "Amplitude",
        "Amplitude",
    ]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )
    
    Xs = [
        Normalized_t,
        Normalized_t,
        Normalized_t,
        np.arange(len(equalizer_coeffs))
    ]
    Ys = [
        {'data': detector_input.reshape(-1), 'label': 'detector_input', 'color': 'red'},
        {'data': error_signal.reshape(-1), 'label': 'error_signal', 'color': 'red'},
        {'data': error_signal_square.reshape(-1), 'label': 'error_signal_square', 'color': 'red'},
        {'data': equalizer_coeffs.reshape(-1), 'label': 'equalizer_coeffs', 'color': 'red'}
    ]
    titles = [
        'detector_input',
        'error_signal',
        'error_signal_square',
        'equalizer_coeffs'
    ]
    xlabels = [
        "Time (t/T)",
        "Time (t/T)",
        "Time (t/T)",
        "equalizer_coeffs idx"
    ]
    ylabels = ["Amplitude"]
    plot_separated(
        Xs=Xs, 
        Ys=Ys, 
        titles=titles,     
        xlabels=xlabels, 
        ylabels=ylabels
    )
        

