import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from Const import RLL_state_machine
from Channel_Modulator import RLL_Modulator
from Channel_Converter import NRZI_Converter
from Disk_Read_Channel import Disk_Read_Channel
from Target_PR_Channel import Target_PR_Channel
from Utils import plot_separated
from Params import Params
sys.path.pop()

class Adaptive_Equalizer(object):
    
    def __init__(self, equalizer_input, reference_signal, taps_num, mu):
        self.equalizer_input = equalizer_input
        self.reference_signal = reference_signal
        self.taps_num = taps_num
        self.equalizer_coeffs = np.zeros((1, self.taps_num))
        self.mu = mu
        self.len_padding = taps_num - 1
        
        print('\nLen Padding in adaptive equalizer training is')
        print(self.len_padding)
        
    def lms(self):   
        equalizer_output = np.zeros(self.equalizer_input.shape)
        error_signal = np.zeros(self.equalizer_input.shape)
        error_signal_square = np.zeros(self.equalizer_input.shape)
        
        equalizer_input = np.pad(self.equalizer_input[:,:], ((0, 0), (self.len_padding, 0)), mode='constant')

        for pos in range(self.equalizer_input.shape[1]):
            equalizer_input_truncation = equalizer_input[:, pos:pos + self.taps_num]
            equalizer_input_truncation = np.fliplr(equalizer_input_truncation)
            equalizer_output[0, pos] = np.dot(self.equalizer_coeffs[0,:], equalizer_input_truncation[0, :])
            error_signal[0, pos] = equalizer_output[0, pos] - self.reference_signal[0, pos]
            self.equalizer_coeffs -= 2 * self.mu * error_signal[0, pos] * equalizer_input_truncation[0, :] 
            error_signal_square[0, pos] = error_signal[0, pos]**2

        return equalizer_output, error_signal, error_signal_square, self.equalizer_coeffs
    
    def equalized_signal(self):
        equalizer_output = np.zeros(self.equalizer_input.shape)
        
        equalizer_output = (np.convolve(self.equalizer_coeffs[0,:], self.equalizer_input[0, :])
               [:-self.len_padding].reshape(self.equalizer_input.shape))
            
        return equalizer_output

if __name__ == '__main__':  

    # constant and input paras
    params = Params()
    encoder_dict, encoder_definite = RLL_state_machine()
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    codeword_len = int(params.equalizer_train_len/rate_constrain)
    
    # class
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel(params)
    target_pr_channel = Target_PR_Channel(params)
    
    Normalized_t = np.linspace(1, int(params.equalizer_train_len/rate_constrain), int(params.equalizer_train_len/rate_constrain))
        
    train_bits = np.random.randint(2, size = (1, params.equalizer_train_len))
    codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(train_bits))
    
    rf_signal = disk_read_channel.RF_signal(codeword)
    equalizer_input = disk_read_channel.awgn(rf_signal, params.snr_train)
    equalizer_input = disk_read_channel.jitter(equalizer_input, params.zeta)
    
    pr_signal = target_pr_channel.target_channel(codeword)
    
    pr_adaptive_equalizer = Adaptive_Equalizer(        
        equalizer_input  = equalizer_input,
        reference_signal = pr_signal,
        taps_num = 15,
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
        {'data': equalizer_input.reshape(-1), 'label': f'equalizer_input_snr{params.snr_train}', 'color': 'red'},
        {'data': pr_signal.reshape(-1), 'label': 'pr_signal', 'color': 'red'},
    ]
    titles = [
        'Binary Sequence',
        'rf_signal',
        f'equalizer_input_snr{params.snr_train}',
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
        np.arange(equalizer_coeffs.shape[1])
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
    
    # save equalizer_coeffs to txt
    if not os.path.exists(params.equalizer_coeffs_dir):
        os.makedirs(params.equalizer_coeffs_dir)
    np.savetxt(params.equalizer_coeffs_file, pr_adaptive_equalizer.equalizer_coeffs)
    print(f"save equalizer_coeffs to txt files:{params.equalizer_coeffs_file}")
    print(f"equalizer_coeffs are {pr_adaptive_equalizer.equalizer_coeffs}")

    # validate  
    info_len = int((params.num_plots*params.eval_length + params.overlap_length)*rate_constrain)
    info = np.random.randint(2, size = (1, info_len))
    codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
    
    rf_signal = disk_read_channel.RF_signal(codeword)
    
    miu = (params.snr_start + params.snr_stop)/2
    sigma = (params.snr_stop - miu)/2
    random_snr = np.random.normal(miu, sigma)
    random_snr = min(max(random_snr, params.snr_start), params.snr_stop)
    
    equalizer_input = disk_read_channel.awgn(rf_signal, random_snr)
    equalizer_input = disk_read_channel.jitter(equalizer_input, params.zeta)
    pr_signal = target_pr_channel.target_channel(codeword)
    
    length = equalizer_input.shape[1]
    
    # actually equalizer output stream data
    pr_adaptive_equalizer.equalizer_input = equalizer_input
    equalizer_output = pr_adaptive_equalizer.equalized_signal()
    
    for pos in range(0, length - params.overlap_length, params.eval_length):
        
        codeword_truncation = codeword[:, pos:pos+params.eval_length+params.overlap_length]
        rf_signal_truncation = rf_signal[:, pos:pos+params.eval_length+params.overlap_length]
        equalizer_input_truncation = equalizer_input[:, pos:pos+params.eval_length+params.overlap_length]
        pr_signal_truncation = pr_signal[:, pos:pos+params.eval_length+params.overlap_length]
        detector_input = equalizer_output[:, pos:pos+params.eval_length+params.overlap_length]
        
        # pr_adaptive_equalizer.equalizer_input  = equalizer_input_truncation
        # pr_adaptive_equalizer.reference_signal = pr_signal_truncation
        # detector_input_train, error_signal, error_signal_square, equalizer_coeffs = pr_adaptive_equalizer.lms()
        
        Normalized_t = np.linspace(1, params.eval_length+params.overlap_length, params.eval_length+params.overlap_length)
        Xs = [
            Normalized_t,
            Normalized_t,
            Normalized_t,
            Normalized_t,
            Normalized_t
        ]
        Ys = [
            {'data': codeword_truncation.reshape(-1), 'label': 'binary Sequence'}, 
            {'data': rf_signal_truncation.reshape(-1), 'label': 'rf_signal_truncation', 'color': 'red'},
            {'data': equalizer_input_truncation.reshape(-1), 'label': f'equalizer_input_truncation_snr{random_snr}', 'color': 'red'},
            {'data': pr_signal_truncation.reshape(-1), 'label': 'pr_signal_truncation', 'color': 'red'},
            {'data': detector_input.reshape(-1), 'label': 'detector_input', 'color': 'red'},
        ]
        titles = [
            'codeword_truncation',
            'rf_signal_truncation',
            f'equalizer_input_truncation_snr{random_snr}',
            'pr_signal_truncation',
            'detector_input',
        ]
        xlabels = ["Time (t/T)"]
        ylabels = [
            "Binary",
            "Amplitude",
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
        
        # Xs = [
        #     Normalized_t,
        #     Normalized_t,
        #     Normalized_t,
        #     np.arange(equalizer_coeffs.shape[1])
        # ]
        # Ys = [
        #     {'data': detector_input_train.reshape(-1), 'label': 'detector_input_train', 'color': 'red'},
        #     {'data': error_signal.reshape(-1), 'label': 'error_signal', 'color': 'red'},
        #     {'data': error_signal_square.reshape(-1), 'label': 'error_signal_square', 'color': 'red'},
        #     {'data': equalizer_coeffs.reshape(-1), 'label': 'equalizer_coeffs', 'color': 'red'}
        # ]
        # titles = [
        #     'detector_input_train',
        #     'error_signal',
        #     'error_signal_square',
        #     'equalizer_coeffs'
        # ]
        # xlabels = [
        #     "Time (t/T)",
        #     "Time (t/T)",
        #     "Time (t/T)",
        #     "equalizer_coeffs idx"
        # ]
        # ylabels = ["Amplitude"]
        # plot_separated(
        #     Xs=Xs, 
        #     Ys=Ys, 
        #     titles=titles,     
        #     xlabels=xlabels, 
        #     ylabels=ylabels
        # )
        

