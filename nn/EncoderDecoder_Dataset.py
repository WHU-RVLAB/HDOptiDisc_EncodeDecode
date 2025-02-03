import os
import numpy as np
import sys
import torch
from torch.utils.data import Dataset
np.set_printoptions(threshold=sys.maxsize)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Const import RLL_state_machine, Target_channel_state_machine
from lib.Utils import sliding_shape
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Params import Params
sys.path.pop()

class PthDataset(Dataset):
    def __init__(self, file_path):
        data = torch.load(file_path, weights_only=False)
        self.data = torch.from_numpy(data['data']).float()
        self.label = torch.from_numpy(data['label']).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :, :], self.label[idx, :]
    
## Rawdb: generate rawdb for neural network
class Rawdb(object):
    def __init__(self, params:Params, encoder_dict, encoder_definite, channel_dict):
        self.params = params
        
        self.encoder_dict = encoder_dict
        self.num_state = len(self.encoder_dict)
        self.num_input_sym_enc = self.encoder_dict[1]['input'].shape[1]
        self.num_out_sym = self.encoder_dict[1]['output'].shape[1]
        self.code_rate = self.num_input_sym_enc / self.num_out_sym
        
        self.channel_dict = channel_dict
        self.ini_state_channel = self.channel_dict['ini_state']
        self.num_input_sym_channel = int(self.channel_dict['in_out'].shape[1]/2)
        
        self.RLL_modulator = RLL_Modulator(encoder_dict, encoder_definite)
        self.NRZI_converter = NRZI_Converter()
        self.disk_read_channel = Disk_Read_Channel(params)
    
    def data_generation(self, prob, info_len):
        '''
        training/testing data(with sliding window) and label
        output: numpy array 
        '''
        params = self.params
        dummy_len = int(params.overlap_length * self.code_rate)
        codeword_len = int(info_len/self.code_rate)
        num_ber = int((params.snr_stop-params.snr_start)/params.snr_step+1)
        
        bt_size_snr = int(codeword_len/params.eval_length)
        bt_size = num_ber*bt_size_snr
        block_length = params.eval_length + params.overlap_length
        data, label = (np.zeros((bt_size, block_length)), np.zeros((bt_size, block_length)))
        
        # generate data and label from stream data
        for snr_idx in np.arange(0, num_ber):
            snr = params.snr_start+snr_idx*params.snr_step
            
            info = np.random.choice(np.arange(0, 2), size = (1, info_len + dummy_len), p=[1-prob, prob])
            
            codeword = self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info))
            rf_signal = self.disk_read_channel.RF_signal(codeword)
            equalizer_input = self.disk_read_channel.awgn(rf_signal, snr)
            
            length = equalizer_input.shape[1]
            for signal_idx, pos in enumerate(range(0, length - params.overlap_length, params.eval_length)):
                
                codeword_truncation = codeword[:, pos:pos+params.eval_length+params.overlap_length]
                rf_signal_truncation = rf_signal[:, pos:pos+params.eval_length+params.overlap_length]
                equalizer_input_truncation = equalizer_input[:, pos:pos+params.eval_length+params.overlap_length]
                
                label[snr_idx*bt_size_snr + signal_idx:snr_idx*bt_size_snr + signal_idx + 1, :] = codeword_truncation
                data[snr_idx*bt_size_snr + signal_idx:snr_idx*bt_size_snr + signal_idx + 1, :]  = equalizer_input_truncation
        
        data = sliding_shape(data, params.input_size)
        label = label
        
        print("generate training/testing data(with sliding window) and label")
        
        return data, label
    
    def data_generation_eval(self, prob, snr):
        '''
        evaluation data (with sliding window) and label
        output: numpy array data_eval, numpy array label_eval
        '''
        params = self.params
        info_len = params.data_val_len
        dummy_len = int(params.overlap_length * self.code_rate)
        codeword_len = int(info_len/self.code_rate)
        num_ber = 1
        
        bt_size_snr = int(codeword_len/params.eval_length)
        bt_size = num_ber*bt_size_snr
        block_length = params.eval_length + params.overlap_length
        data, label = (np.zeros((bt_size, block_length)), np.zeros((bt_size, block_length)))
        
        # generate data and label from stream data
        info = np.random.choice(np.arange(0, 2), size = (1, info_len + dummy_len), p=[1-prob, prob])
        
        codeword = self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info))
        rf_signal = self.disk_read_channel.RF_signal(codeword)
        equalizer_input = self.disk_read_channel.awgn(rf_signal, snr)
        
        length = equalizer_input.shape[1]
        for signal_idx, pos in enumerate(range(0, length - params.overlap_length, params.eval_length)):
            
            codeword_truncation = codeword[:, pos:pos+params.eval_length+params.overlap_length]
            rf_signal_truncation = rf_signal[:, pos:pos+params.eval_length+params.overlap_length]
            equalizer_input_truncation = equalizer_input[:, pos:pos+params.eval_length+params.overlap_length]
            
            label[signal_idx:signal_idx + 1, :] = codeword_truncation
            data[signal_idx:signal_idx + 1, :]  = equalizer_input_truncation
        
        data = sliding_shape(data, params.input_size)
        label = label
        
        print("generate evaluation data (with sliding window) and label")
        
        return data, label
    
    def build_rawdb(self, data_dir):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        params =  self.params
        block_length = params.eval_length + params.overlap_length

        data = np.empty((0, block_length, params.input_size))
        label = np.empty((0, block_length))
        for _ in range(params.train_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_train, label_train = self.data_generation(random_p, params.data_train_len)
            data = np.append(data, data_train, axis=0)
            label = np.append(label, label_train, axis=0)

        file_path = f"{data_dir}/classifier_train_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path)
        print("generate training dataset\n")

        data = np.empty((0, block_length, params.input_size))
        label = np.empty((0, block_length))
        for _ in range(params.test_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_test, label_test = self.data_generation(random_p, params.data_test_len)
            data = np.append(data, data_test, axis=0)
            label = np.append(label, label_test, axis=0)

        file_path = f"{data_dir}/classifier_test_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path)
        print("generate testing dataset\n")

        data = np.empty((0, block_length, params.input_size))
        label = np.empty((0, block_length))
        for _ in range(params.validate_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            miu = (params.snr_start + params.snr_stop)/2
            sigma = (params.snr_stop - miu)/2
            random_snr = np.random.normal(miu, sigma)
            random_snr = min(max(random_snr, params.snr_start), params.snr_stop)

            data_val, label_val = self.data_generation_eval(random_p, random_snr)
            data = np.append(data, data_val, axis=0)
            label = np.append(label, label_val, axis=0)

        file_path = f"{data_dir}/classifier_validate_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path)
        print("generate validate dataset\n")

if __name__ == '__main__':
    params = Params()

    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()

    rawdb = Rawdb(params, encoder_dict, encoder_definite, channel_dict)

    rawdb.build_rawdb("../data")