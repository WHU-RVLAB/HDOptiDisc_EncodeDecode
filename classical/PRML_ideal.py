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
from lib.Target_PR_Channel import Target_PR_Channel
sys.path.pop()
from algorithm.Viterbi import Viterbi

def ideal_sys(params:Params):
    
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
    target_pr_channel = Target_PR_Channel(params)
    viterbi_detector = Viterbi(params, channel_dict, ini_metric)

    # define ber
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step+1)
    codeword_len = int(params.eval_info_len/rate_constrain)
    
    # eval mode
    ber_list = []
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.eval_info_len + dummy_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        
        signal_upsample_ideal, signal_upsample_jittered, pr_signal_ideal, pr_signal_real = target_pr_channel.target_channel_jitter(codeword)
        if params.jitteron:
           pr_signal_input = pr_signal_real
        else:
           pr_signal_input = pr_signal_ideal
        pr_signal_noise = target_pr_channel.awgn(pr_signal_input, snr)
        
        detectword = np.empty((1, 0))
        length = pr_signal_noise.shape[1]
        for pos in range(0, length - params.post_overlap_length, params.eval_length):
            pr_signal_noise_truncation = pr_signal_noise[:, pos:pos+params.eval_length+params.post_overlap_length]
            dec_tmp, metric_next = viterbi_detector.vit_dec(pr_signal_noise_truncation, ini_metric)
            ini_metric = metric_next
            detectword = np.append(detectword, dec_tmp, axis=1)
        
        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - detectword[:, 0:codeword_len])) 
               / codeword_len)
        print("The bit error rate (BER) is:")
        print(ber)
        ber_list.append(ber)

    if params.jitteron == True and params.addsineon == True:
        ber_file = "../data/PRMLideal_jitter_addsine_result.txt"
    elif params.jitteron == True and params.addsineon == False:
        ber_file = "../data/PRMLideal_jitter_result.txt"
    elif params.jitteron == False and params.addsineon == True:
        ber_file = "../data/PRMLideal_addsine_result.txt"
    elif params.jitteron == False and params.addsineon == False:
        ber_file = "../data/PRMLideal_result.txt"
        
    with open(ber_file, "w") as file:
        for ber in ber_list:
            file.write(f"{ber}\n")
    print(f"ber data have save to {ber_file}")
        
if __name__ == '__main__':
    params = Params()
    ideal_sys(params)