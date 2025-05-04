import sys
import os
import numpy as np
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Const import RLL_state_machine, Target_channel_state_machine
from lib.Params import Params
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Adaptive_Equalizer import Adaptive_Equalizer
sys.path.pop()
from algorithm.Viterbi_NP import Viterbi_NP
from algorithm.Noise_Predictor import Noise_Predictor

def realistic_sys(params:Params):
    
    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()
    if params.signal_norm:
        channel_dict['in_out'][:, 1] /= sum(params.PR_coefs)

    # Initial metric 
    ini_metric = 1000 * np.ones((channel_dict['num_state'], 1))
    ini_metric[0, 0] = 0
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    dummy_len = int(params.post_overlap_length * num_sym_in_constrain 
                 / num_sym_out_constrain)
    
    # class
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel(params)
    noise_predictor = Noise_Predictor(params)
    viterbi_np_detector = Viterbi_NP(params, channel_dict, ini_metric)
    
    pr_adaptive_equalizer = Adaptive_Equalizer(        
        equalizer_input  = None,
        reference_signal = None,
        taps_num = params.equalizer_taps_num,
        mu = params.equalizer_mu
    )

    if params.jitteron == True and params.addsineon == True:
        equalizer_load_file = params.equalizer_coeffs_jitter_sine_file
    elif params.jitteron == True and params.addsineon == False:
        equalizer_load_file = params.equalizer_coeffs_jitter_file
    elif params.jitteron == False and params.addsineon == True:
        equalizer_load_file = params.equalizer_coeffs_sine_file
    elif params.jitteron == False and params.addsineon == False:
        equalizer_load_file = params.equalizer_coeffs_file
        
    pr_adaptive_equalizer.equalizer_coeffs = np.loadtxt(equalizer_load_file).reshape(1, -1)
    print(f"\nload equalizer_coeffs from txt files:{equalizer_load_file}")
    print(f"\nequalizer_coeffs are {pr_adaptive_equalizer.equalizer_coeffs}")
    
    equalizer_coeffs = pr_adaptive_equalizer.equalizer_coeffs
    coeff_num_side = int((equalizer_coeffs.shape[1] - 1)/2)
    equalizer_coeffs_rot = np.append(equalizer_coeffs[:, -coeff_num_side:], 
                        equalizer_coeffs[:, :coeff_num_side+1], axis=1)
    
    # define ber
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step+1)
    codeword_len = int(params.eval_info_len/rate_constrain)
    
    # eval mode
    ber_list = []
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.eval_info_len + dummy_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        
        signal_upsample_ideal, signal_upsample_jittered, rf_signal_ideal, rf_signal = disk_read_channel.RF_signal_jitter(codeword)

        if params.jitteron:
            rf_signal_input = rf_signal
        else:
            rf_signal_input = rf_signal_ideal

        equalizer_input = disk_read_channel.awgn(rf_signal_input, snr)

        if params.addsineon:
            equalizer_input = disk_read_channel.addsin(equalizer_input)
        
        # actually equalizer output stream data
        pr_adaptive_equalizer.equalizer_input = equalizer_input
        equalizer_output = pr_adaptive_equalizer.equalized_signal()
        detector_input = equalizer_output
        
        pred_coef, mmse = noise_predictor.predictor(equalizer_coeffs_rot, snr)
        detectword = viterbi_np_detector.dec(detector_input, pred_coef)
        
        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - detectword[:, 0:codeword_len])) 
               / codeword_len)
        print("The bit error rate (BER) is:")
        print(ber)
        ber_list.append(ber)

    if params.jitteron == True and params.addsineon == True:
        ber_file = "../data/NPML_jitter_addsine_result.txt"
    elif params.jitteron == True and params.addsineon == False:
        ber_file = "../data/NPML_jitter_result.txt"
    elif params.jitteron == False and params.addsineon == True:
        ber_file = "../data/NPML_addsine_result.txt"
    elif params.jitteron == False and params.addsineon == False:
        ber_file = "../data/NPML_result.txt"
        
    with open(ber_file, "w") as file:
        for ber in ber_list:
            file.write(f"{ber}\n")
    print(f"ber data have save to {ber_file}")
        
if __name__ == '__main__':
    params = Params()
    realistic_sys(params)