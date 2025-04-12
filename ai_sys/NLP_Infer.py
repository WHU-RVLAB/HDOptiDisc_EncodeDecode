import numpy as np
import torch
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

from NLP.BaseModel import BaseModel
from NLP.RNN import RNN
from NLP.RNNScratch import RNNScratch
from NLP.Transformer import Transformer
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Const import RLL_state_machine, Target_channel_state_machine
from lib.Utils import sliding_shape
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Params import Params
sys.path.pop()

np.random.seed(12345)

def ai_nlp_sys():
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
    dummy_len = int(params.overlap_length * num_sym_in_constrain 
                 / num_sym_out_constrain)
    
    # class
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel(params)

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model
    model_file = None
    if params.model_arch == "rnn":
        model = RNN(params, device).to(device)
        model_file = "nlp_rnn.pth.tar"
    elif params.model_arch == "rnn_scratch":
        model = RNNScratch(params, device).to(device)
        model_file = "nlp_rnn_scratch.pth.tar"
    elif params.model_arch == "transformer":
        model = Transformer(params, device).to(device)
        model_file = "nlp_transformer.pth.tar"

    # load model from model_file
    model_path = f"{params.model_dir}/{model_file}"
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
    
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
        if params.model_arch == "rnn" or params.model_arch == "rnn_scratch":
            hidden_dim = params.rnn_hidden_size
            num_layers = 2*params.rnn_layer
        elif params.model_arch == "transformer":
            hidden_dim = params.transformer_hidden_size
            num_layers = params.transformer_decoder_layers  
        init_hidden = torch.zeros(num_layers, 1, hidden_dim, device=device)
        for pos in range(0, length - params.overlap_length, params.eval_length):
            equalizer_input_truncation = equalizer_input[:, pos:pos+params.eval_length+params.overlap_length]
            
            truncation_input = sliding_shape(equalizer_input_truncation, params.nlp_input_size)
            truncation_input = torch.from_numpy(truncation_input).float().to(device)
            dec_tmp, init_hidden = model.decode(params.eval_length, truncation_input, init_hidden, device)
            init_hidden = torch.zeros(num_layers, 1, hidden_dim, device=device)
            
            decodeword = np.append(decodeword, dec_tmp, axis=1)

        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - decodeword[:, 0:codeword_len])) / codeword_len)
        print(f"The bit error rate (BER) use {params.model_arch} is:")
        print(ber)
        ber_list.append(ber)
    
    ber_file = f"../data/nlp_{params.model_arch}_result.txt"
    with open(ber_file, "w") as file:
        for ber in ber_list:
            file.write(f"{ber}\n")
    print(f"ber data have save to {ber_file}")

if __name__ == '__main__':
    ai_nlp_sys()