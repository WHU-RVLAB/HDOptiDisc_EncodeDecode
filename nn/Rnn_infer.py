import argparse
import numpy as np
import torch
import sys
import os
sys.path.append("..")
np.set_printoptions(threshold=sys.maxsize)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Const import RLL_state_machine, Target_channel_state_machine, Target_channel_dummy_bits
from lib.Utils import sliding_shape, evaluation
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Target_PR_Channel import Target_PR_Channel
sys.path.pop()

from Rnn_train import RNN

np.random.seed(12345)

parser = argparse.ArgumentParser()

parser.add_argument('-info_len', type=int, default=1000000)

parser.add_argument('-snr_start', type=float, default=40)
parser.add_argument('-snr_stop', type=float, default=60)
parser.add_argument('-snr_step', type=float, default=10)

parser.add_argument('-eval_length', type=int, default=30)
parser.add_argument('-overlap_length', type=int, default=30)

parser.add_argument('-input_size', type=int, default=5)
parser.add_argument('-rnn_input_size', type=int, default=5)
parser.add_argument('-rnn_hidden_size', type=int, default=50)
parser.add_argument('-output_size', type=int, default=1)
parser.add_argument('-rnn_layer', type=int, default=10)
parser.add_argument('-rnn_dropout_ratio', type=float, default=0)

parser.add_argument('-model_file', default="../model/model.pth.tar", type=str, metavar='PATH', help='path to latest model')

global args
args = parser.parse_args()

def rnn_sys():
    
    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()
    dummy_start_paths, dummy_start_input, dummy_start_output, dummy_start_eval, \
    dummy_end_paths, dummy_end_input, dummy_end_output, dummy_end_eval = Target_channel_dummy_bits()
    
    # rate for constrained code
    num_sym_in_constrain = encoder_dict[1]['input'].shape[1]
    num_sym_out_constrain = encoder_dict[1]['output'].shape[1]
    rate_constrain = num_sym_in_constrain / num_sym_out_constrain
    dummy_len = int(args.overlap_length * num_sym_in_constrain / num_sym_out_constrain)
    
    # class
    RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
    NRZI_converter = NRZI_Converter()
    disk_read_channel = Disk_Read_Channel()

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = RNN(args, device).to(device)

    # load model from model_file
    if args.model_file:
        if os.path.isfile(args.model_file):
            print("=> loading checkpoint '{}'".format(args.model_file))
            checkpoint = torch.load(args.model_file, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.model_file))
    
    # define ber
    num_ber = int((args.snr_stop-args.snr_start)/args.snr_step+1)
    codeword_len = int(args.info_len+dummy_len)
    ber_channel = np.zeros((1, num_ber))
    ber_info = np.zeros((1, num_ber))
    
    # eval RNN
    for idx in np.arange(0, num_ber):
        snr = args.snr_start+idx*args.snr_step
        
        info = np.random.randint(2, size = (1, args.info_len+dummy_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        
        rf_signal = disk_read_channel.RF_signal(codeword)
        equalizer_input = disk_read_channel.awgn(rf_signal, snr)
        
        length = equalizer_input.shape[1]
        decodeword = np.empty((1, 0))
        for pos in range(0, length - args.overlap_length, args.eval_length):
            equalizer_input_truncation = equalizer_input[:, pos:pos+args.eval_length+args.overlap_length]
            truncation_input = sliding_shape(equalizer_input_truncation, args.input_size)
            truncation_input = torch.from_numpy(truncation_input).float().to(device)
            dec_tmp = evaluation(args.eval_length, truncation_input, model, device)
            decodeword = np.append(decodeword, dec_tmp, axis=1)

        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - decodeword[:, 0:codeword_len])) / codeword_len)
        print("The bit error rate (BER) use RNN is:")
        print(ber)

if __name__ == '__main__':
    rnn_sys()