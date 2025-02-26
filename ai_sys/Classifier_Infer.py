import numpy as np
import torch
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

from LR import LR
from XGBoost import XGBoost
from MLP import MLP
from CNN import CNN
from Unet1D import UNet1D
from RNN import RNN
from Transformer import Transformer
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

def ai_sys():
    global params
    params = Params()
    
    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()
    
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
    is_nn = 0
    if params.model_arch == "lr":
        model = LR(params)
        model_file = "lr_model.joblib"
    elif params.model_arch == "xgboost":
        model = XGBoost(params)
        model_file = "xgb_model.json"
    elif params.model_arch == "mlp":
        is_nn = 1
        model = MLP(params, device).to(device)
        model_file = "mlp.pth.tar"
    elif params.model_arch == "cnn":
        is_nn = 1
        model = CNN(params, device).to(device)
        model_file = "cnn.pth.tar"
    elif params.model_arch == "unet":
        is_nn = 1
        model = UNet1D(params, device).to(device)
        model_file = "unet.pth.tar"
    elif params.model_arch == "rnn":
        is_nn = 1
        model = RNN(params, device).to(device)
        model_file = "rnn.pth.tar"
    elif params.model_arch == "transformer":
        is_nn = 1
        model = Transformer(params, device).to(device)
        model_file = "transformer.pth.tar"

    # load model from model_file
    model_path = f"{params.model_dir}/{model_file}"
    if is_nn:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
    else:
        model.load_model(model_path)
    
    # define ber
    num_ber = int((params.snr_stop-params.snr_start)/params.snr_step+1)
    codeword_len = int(params.eval_info_len/rate_constrain)
    ber_channel = np.zeros((1, num_ber))
    ber_info = np.zeros((1, num_ber))
    
    # eval AI sys
    ber_list = []
    for idx in np.arange(0, num_ber):
        snr = params.snr_start+idx*params.snr_step
        
        info = np.random.randint(2, size = (1, params.eval_info_len + dummy_len))
        codeword = NRZI_converter.forward_coding(RLL_modulator.forward_coding(info))
        
        rf_signal = disk_read_channel.RF_signal(codeword)
        equalizer_input = disk_read_channel.awgn(rf_signal, snr)
        equalizer_input = disk_read_channel.jitter(equalizer_input, params.zeta)
        
        length = equalizer_input.shape[1]
        decodeword = np.empty((1, 0))
        for pos in range(0, length - params.overlap_length, params.eval_length):
            equalizer_input_truncation = equalizer_input[:, pos:pos+params.eval_length+params.overlap_length]
            truncation_input = sliding_shape(equalizer_input_truncation, params.input_size)
            if is_nn:
                truncation_input = torch.from_numpy(truncation_input).float().to(device)
                dec_tmp = model.decode(params.eval_length, truncation_input, device)
            else:
                dec_tmp = model.decode(params.eval_length, truncation_input[0, :, :])
            decodeword = np.append(decodeword, dec_tmp, axis=1)

        print("The SNR is:")
        print(snr)
        ber = (np.count_nonzero(np.abs(codeword[:, 0:codeword_len] - decodeword[:, 0:codeword_len])) / codeword_len)
        print(f"The bit error rate (BER) use {params.model_arch} is:")
        print(ber)
        ber_list.append(ber)
    
    ber_file = f"../data/{params.model_arch}_result.txt"
    with open(ber_file, "w") as file:
        for ber in ber_list:
            file.write(f"{ber}\n")
    print(f"ber data have save to {ber_file}")

if __name__ == '__main__':
    ai_sys()