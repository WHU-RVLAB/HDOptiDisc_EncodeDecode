import os
import numpy as np
import math
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
    
    def data_generation_train(self, prob, bt_size_snr):
        '''
        training/testing data(with sliding window) and label
        output: numpy array 
        '''
        
        bt_size = int(((self.params.snr_stop-self.params.snr_start)/
                         self.params.snr_step+1)*bt_size_snr)
        
        block_length = self.params.eval_length + self.params.overlap_length
        info_length = math.ceil(block_length/self.num_out_sym)*self.num_input_sym_enc
        
        info = np.random.choice(np.arange(0, 2), size = (bt_size_snr, info_length), p=[1-prob, prob])
        
        data, label = (np.zeros((bt_size, block_length)), 
                                 np.zeros((bt_size, block_length)))
                
        for i in range(bt_size_snr):
            codeword = (self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info[i : i+1, :]))[:, :block_length])
            
            for idx in np.arange(0, (self.params.snr_stop-self.params.snr_start)/self.params.snr_step+1):
                label[int(idx*bt_size_snr+i) : int(idx*bt_size_snr+i+1), :] = codeword
                
                rf_signal = self.disk_read_channel.RF_signal(codeword)
                equalizer_input = self.disk_read_channel.awgn(rf_signal, self.params.snr_start+idx*self.params.snr_step)
                
                data[int(idx*bt_size_snr+i) : int(idx*bt_size_snr+i+1), :] = equalizer_input
        
        data = sliding_shape(data, self.params.input_size)
        label = label
        
        return data, label
    
    def data_generation_eval(self, prob, snr):
        '''
        evaluation data(without sliding window) and label
        output: numpy array data_eval, numpy array label_eval
        '''
        code_rate = 2/3
        info = np.random.choice(np.arange(0, 2), size = (1, int(code_rate * self.params.batch_size_val)),  p=[1-prob, prob])
        codeword = self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info))
        
        rf_signal = self.disk_read_channel.RF_signal(codeword)
        equalizer_input = self.disk_read_channel.awgn(rf_signal, snr)

        equalizer_input = np.concatenate((equalizer_input, (np.zeros((1, self.params.overlap_length)))), axis=1)

        block_length = self.params.eval_length + self.params.overlap_length
        sliding_equalizer_input = np.empty((0, block_length))
        x_len = equalizer_input.shape[1]
        for idx in range(block_length, x_len + 1, self.params.eval_length):
            block = equalizer_input[:, idx-block_length : idx]
            sliding_equalizer_input = np.append(sliding_equalizer_input, block, axis=0)

        data = sliding_shape(sliding_equalizer_input, self.params.input_size)
        label = codeword.reshape(-1, self.params.eval_length)
        
        return data, label
    
    def build_rawdb(self, data_dir):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        block_length = self.params.eval_length + self.params.overlap_length

        data = np.empty((0, block_length, self.params.input_size))
        label = np.empty((0, block_length))
        for idx in range(self.params.train_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_train, label_train = self.data_generation_train(random_p, self.params.batch_size_train)
            data = np.append(data, data_train, axis=0)
            label = np.append(label, label_train, axis=0)

        file_path = f"{data_dir}/train_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path)

        data = np.empty((0, block_length, self.params.input_size))
        label = np.empty((0, block_length))
        for idx in range(self.params.test_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_test, label_test = self.data_generation_train(random_p, self.params.batch_size_test)
            data = np.append(data, data_test, axis=0)
            label = np.append(label, label_test, axis=0)

        file_path = f"{data_dir}/test_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path)

        data = np.empty((0, block_length, self.params.input_size))
        label = np.empty((0, self.params.eval_length))
        for idx in range(self.params.validate_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            miu = (self.params.snr_start + self.params.snr_stop)/2
            sigma = (self.params.snr_stop - miu)/2
            random_snr = np.random.normal(miu, sigma)
            random_snr = min(max(random_snr, self.params.snr_start), self.params.snr_stop)

            data_val, label_val = self.data_generation_eval(random_p, random_snr)
            data = np.append(data, data_val, axis=0)
            label = np.append(label, label_val, axis=0)

        file_path = f"{data_dir}/validate_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path)

if __name__ == '__main__':
    params = Params()

    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()

    rawdb = Rawdb(params, encoder_dict, encoder_definite, channel_dict)

    rawdb.build_rawdb("../data")