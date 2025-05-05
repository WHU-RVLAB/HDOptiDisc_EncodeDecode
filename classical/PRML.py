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
from algorithm.Viterbi import Viterbi

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
    viterbi_detector = Viterbi(params, channel_dict, ini_metric)
    
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
        
        length = equalizer_input.shape[1]
        
        # actually equalizer output stream data
        pr_adaptive_equalizer.equalizer_input = equalizer_input
        equalizer_output = pr_adaptive_equalizer.equalized_signal()
        detectword = np.empty((1, 0))
        for pos in range(0, length - params.post_overlap_length, params.eval_length):
            detector_input = equalizer_output[:, pos:pos+params.eval_length+params.post_overlap_length]
            dec_tmp, metric_next = viterbi_detector.vit_dec(detector_input, ini_metric)
            ini_metric = metric_next
            detectword = np.append(detectword, dec_tmp, axis=1)
            
            # # eval mode 
            # # Guarantees that the equalizer can track changes in channel characteristics
            # pr_adaptive_equalizer.equalizer_input  = equalizer_input_truncation
            # pr_adaptive_equalizer.reference_signal = pr_signal_truncation
            # detector_input_train, error_signal, error_signal_square, equalizer_coeffs = pr_adaptive_equalizer.lms()
            
            # Normalized_t = np.linspace(0, params.eval_length+params.post_overlap_length -1, params.eval_length+params.post_overlap_length)
            # Xs = [
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t,
            #     Normalized_t
            # ]
            # Ys = [
            #     {'data': codeword_truncation.reshape(-1), 'label': 'binary Sequence'}, 
            #     {'data': rf_signal_truncation.reshape(-1), 'label': 'rf_signal_truncation', 'color': 'red'},
            #     {'data': equalizer_input_truncation.reshape(-1), 'label': f'equalizer_input_truncation_snr{snr}', 'color': 'red'},
            #     {'data': pr_signal_truncation.reshape(-1), 'label': 'pr_signal_truncation', 'color': 'red'},
            #     {'data': detector_input.reshape(-1), 'label': 'detector_input', 'color': 'red'},
            #     {'data': pr_signal_noise_truncation.reshape(-1), 'label': 'pr_signal_noise_truncation', 'color': 'red'},
            # ]
            # titles = [
            #     'codeword_truncation',
            #     'rf_signal_truncation',
            #     f'equalizer_input_truncation_snr{snr}',
            #     'pr_signal_truncation',
            #     'detector_input',
            #     'pr_signal_noise_truncation'
            # ]
            # xlabels = ["Time (t/T)"]
            # ylabels = [
            #     "Binary",
            #     "Amplitude",
            #     "Amplitude",
            #     "Amplitude",
            #     "Amplitude",
            #     "Amplitude",
            # ]
            # plot_separated(
            #     Xs=Xs, 
            #     Ys=Ys, 
            #     titles=titles,     
            #     xlabels=xlabels, 
            #     ylabels=ylabels
            # )
            
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
        
        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - detectword[:, 0:codeword_len])) 
               / codeword_len)
        print("The bit error rate (BER) is:")
        print(ber)
        ber_list.append(ber)

    if not os.path.exists(params.algorithm_result_dir):
            os.makedirs(params.algorithm_result_dir)
            
    if params.jitteron == True and params.addsineon == True:
        ber_file = f"{params.algorithm_result_dir}/PRML_jitter_addsine_result.txt"
    elif params.jitteron == True and params.addsineon == False:
        ber_file = f"{params.algorithm_result_dir}/PRML_jitter_result.txt"
    elif params.jitteron == False and params.addsineon == True:
        ber_file = f"{params.algorithm_result_dir}/PRML_addsine_result.txt"
    elif params.jitteron == False and params.addsineon == False:
        ber_file = f"{params.algorithm_result_dir}/PRML_result.txt"
        
    with open(ber_file, "w") as file:
        for ber in ber_list:
            file.write(f"{ber}\n")
    print(f"ber data have save to {ber_file}")
        
if __name__ == '__main__':
    params = Params()
    realistic_sys(params)