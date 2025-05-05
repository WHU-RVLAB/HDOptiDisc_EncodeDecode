import numpy as np
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

from NLP.RNN import RNN
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))
from lib.Const import RLL_state_machine, Target_channel_state_machine
from lib.Utils import sliding_shape
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Params import Params
sys.path.pop()

def ai_nlp_deploy_sys():
    global params
    params = Params()
    
    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()
    if params.signal_norm:
        channel_dict['in_out'][:, 1] /= sum(params.PR_coefs)
    
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

    # deploy model
    model = RNN(params)
    
    # define ber
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step+1)
    codeword_len = int(params.eval_info_len/rate_constrain)
    
    # eval AI_NLP sys
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
        
        length = equalizer_input.shape[1]
        decodeword = np.empty((1, 0))
        if params.model_arch == "rnn":
            hidden_dim = params.rnn_hidden_size
            rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
            num_layers = rnn_hidden_size_factor*params.rnn_layer
            
        init_signal =  np.zeros((1, params.pre_overlap_length))
        init_hidden =  np.zeros((num_layers, 1, hidden_dim))
        error_distribution = np.zeros(params.pre_overlap_length+params.eval_length+params.post_overlap_length)
        cur_hidden = init_hidden
        for pos in range(0, length - params.post_overlap_length, params.eval_length):
            if pos:
                equalizer_input_truncation = equalizer_input[:, pos-params.pre_overlap_length:pos+params.eval_length+params.post_overlap_length]
                codeword_truncation = codeword[:, pos-params.pre_overlap_length:pos+params.eval_length+params.post_overlap_length]
            else:
                equalizer_input_truncation = equalizer_input[:, pos:pos+params.eval_length+params.post_overlap_length]
                codeword_truncation = codeword[:, pos:pos+params.eval_length+params.post_overlap_length]
                equalizer_input_truncation = np.concatenate((init_signal, equalizer_input_truncation), axis=1)
                codeword_truncation = np.concatenate((init_signal, codeword_truncation), axis=1)
            
            truncation_input = sliding_shape(equalizer_input_truncation, params.nlp_input_size)
            dec_tmp, nxt_hidden = model.decode(truncation_input, cur_hidden)
            dec_tmp_truncation = dec_tmp[:, params.pre_overlap_length:params.pre_overlap_length + params.eval_length]
            
            # whether transmis hidde state
            if params.nlp_transmis_hidden:
                cur_hidden = nxt_hidden.detach()
            
            codeword_diff = np.abs(codeword_truncation - dec_tmp)
            ber_truncation = (np.count_nonzero(codeword_diff) / (params.pre_overlap_length+params.eval_length+params.post_overlap_length))
            print(f"The ber_truncation use {params.model_arch} is:{ber_truncation}")
            error_indices = np.nonzero(codeword_diff.reshape(-1))[0]
            print(f"Indices with errors: {error_indices}")
            for error_indice in error_indices:
                error_distribution[error_indice] += 1
            
            decodeword = np.append(decodeword, dec_tmp_truncation, axis=1)

        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - decodeword[:, 0:codeword_len])) / codeword_len)
        print(f"The bit error rate (BER) use {params.model_arch} is:")
        print(ber)
        print(f"Error distribution: {error_distribution}")
        ber_list.append(ber)
    
    if not os.path.exists(params.algorithm_result_dir):
        os.makedirs(params.algorithm_result_dir)
        
    ber_file = f"../{params.algorithm_result_dir}/nlp_{params.model_arch}_deploy_result.txt"
    with open(ber_file, "w") as file:
        for ber in ber_list:
            file.write(f"{ber}\n")
    print(f"ber data have save to {ber_file}")

if __name__ == '__main__':
    ai_nlp_deploy_sys()