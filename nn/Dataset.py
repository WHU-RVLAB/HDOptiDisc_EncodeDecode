import os
import argparse
import numpy as np
import math
import sys
import torch
from torch.utils.data import Dataset
import h5py
np.set_printoptions(threshold=sys.maxsize)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Const import RLL_state_machine, Target_channel_state_machine, Target_channel_dummy_bits
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
sys.path.pop()

parser = argparse.ArgumentParser()

parser.add_argument('-eval_info_length', type=int, default=1000000)
parser.add_argument('-eval_length', type=int, default=10)
parser.add_argument('-overlap_length', type=int, default=20)

parser.add_argument('-batch_size_snr_train_1', type=int, default=30)
parser.add_argument('-batch_size_snr_train_2', type=int, default=30)
parser.add_argument('-batch_size_snr_validate_1', type=int, default=600)
parser.add_argument('-batch_size_snr_validate_2', type=int, default=600)
parser.add_argument('-snr_start', type=float, default=5)
parser.add_argument('-snr_stop', type=float, default=40)
parser.add_argument('-snr_step', type=float, default=0.5)

parser.add_argument('-input_size', type=int, default=5)

parser.add_argument('-train_set_batches', type=int, default=10)
parser.add_argument('-test_set_batches', type=int, default=2)
parser.add_argument('-validate_set_batches', type=int, default=2)

class H5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5py.File(h5_file, 'r')
        
        self.data = self.h5_file['data']
        self.label = self.h5_file['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data  = torch.from_numpy(self.data[idx, :, :]).float()
        label = torch.from_numpy(self.label[idx, :]).float()
        
        return data, label

    def __del__(self):
        self.h5_file.close()

## Rawdb: generate rawdb for neural network
class Rawdb(object):
    def __init__(self, args, encoder_dict, encoder_definite, channel_dict):
        self.args = args
        
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
        self.disk_read_channel = Disk_Read_Channel()
    
    def data_generation_train(self, prob, bt_size_snr_1, bt_size_snr_2):
        '''
        training/testing data(with sliding window) and label
        output: numpy array 
        '''
        
        bt_size_1 = int(((args.snr_stop-args.snr_start)/
                         args.snr_step+1)*bt_size_snr_1)
        bt_size_2 = int(((args.snr_stop-args.snr_start)/
                         args.snr_step+1)*bt_size_snr_2)
        
        block_length = args.eval_length + args.overlap_length
        info_length = math.ceil(block_length/self.num_out_sym)*self.num_input_sym_enc
        
        info_1 = np.random.choice(np.arange(0, 2), size = (bt_size_snr_1, info_length), p=[1-prob, prob])
        
        info_2 = np.random.choice(np.arange(0, 2), size = (bt_size_snr_2, info_length), p=[1-prob, prob])
        
        data_1, label_1 = (np.zeros((bt_size_1, block_length)), 
                                 np.zeros((bt_size_1, block_length)))
        data_2, label_2 = (np.zeros((bt_size_2, block_length)), 
                                 np.zeros((bt_size_2, block_length)))
                
        for i in range(bt_size_snr_1):
            codeword = (self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info_1[i : i+1, :]))[:, :block_length])
            
            for idx in np.arange(0, (args.snr_stop-args.snr_start)/args.snr_step+1):
                label_1[int(idx*bt_size_snr_1+i) : int(idx*bt_size_snr_1+i+1), :] = codeword
                
                rf_signal = self.disk_read_channel.RF_signal(codeword)
                equalizer_input = self.disk_read_channel.awgn(rf_signal, args.snr_start+idx*args.snr_step)
                
                data_1[int(idx*bt_size_snr_1+i) : int(idx*bt_size_snr_1+i+1), :] = equalizer_input
        
        for i in range(bt_size_snr_2):
            codeword = (self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info_2[i : i+1, :]))[:, :block_length])
            
            for idx in np.arange(0, (args.snr_stop-args.snr_start)/args.snr_step+1):
                label_2[int(idx*bt_size_snr_2+i) : int(idx*bt_size_snr_2+i+1), :] = codeword
                
                rf_signal = self.disk_read_channel.RF_signal(codeword)
                equalizer_input = self.disk_read_channel.awgn(rf_signal, args.snr_start+idx*args.snr_step)
                
                data_2[int(idx*bt_size_snr_2+i) : int(idx*bt_size_snr_2+i+1), :] = equalizer_input
        
        data = np.append(data_1, data_2, axis=0)
        label = np.append(label_1, label_2, axis=0)
        
        data = self.sliding_shape(data)
        label = label
        
        return data, label
    
    def data_generation_eval(self, prob, snr):
        '''
        evaluation data(without sliding window) and label
        output: numpy array data_eval, numpy array label_eval
        '''
        info = np.random.choice(np.arange(0, 2), size = (1, args.eval_info_length),  p=[1-prob, prob])
        codeword = self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info))
        
        rf_signal = self.disk_read_channel.RF_signal(codeword)
        equalizer_input = self.disk_read_channel.awgn(rf_signal, snr)

        block_length = args.eval_length + args.overlap_length
        equalizer_input = equalizer_input.reshape(-1, block_length)
        codeword = codeword.reshape(-1, block_length)

        data = self.sliding_shape(equalizer_input)
        label = codeword
        
        return data, label
    
    def sliding_shape(self, x):
        '''
        Input: (1, length) numpy array
        Output: (input_size, length) numpy array
        Mapping: sliding window for each time step
        '''
        
        batch_size, time_step = x.shape
        zero_padding_len = args.input_size - 1
        
        x = np.concatenate((np.zeros((batch_size, zero_padding_len)), x), axis=1)
        y = np.zeros((batch_size, time_step, args.input_size))
        
        for bt in range(batch_size):
            for time in range(time_step):
                y[bt, time, :] = x[bt, time:time+args.input_size]
        
        return y.astype(np.float32)
    
    def build_rawdb(self, data_dir):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        block_length = args.eval_length + args.overlap_length

        data = np.empty((0, block_length, args.input_size))
        label = np.empty((0, block_length))
        for idx in range(args.train_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_train, label_train = self.data_generation_train(random_p, args.batch_size_snr_train_1, args.batch_size_snr_train_2)
            data = np.append(data, data_train, axis=0)
            label = np.append(label, label_train, axis=0)

        file_path = f"{data_dir}/train_set.h5"
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('label', data=label)

        data = np.empty((0, block_length, args.input_size))
        label = np.empty((0, block_length))
        for idx in range(args.test_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_test, label_test = self.data_generation_train(random_p, args.batch_size_snr_validate_1, args.batch_size_snr_validate_2)
            data = np.append(data, data_test, axis=0)
            label = np.append(label, label_test, axis=0)

        file_path = f"{data_dir}/test_set.h5"
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('label', data=label)

        data = np.empty((0, block_length, args.input_size))
        label = np.empty((0, block_length))
        for idx in range(args.validate_set_batches):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            miu = (args.snr_start + args.snr_stop)/2
            sigma = (args.snr_stop - miu)/2
            random_snr = np.random.normal(miu, sigma)
            random_snr = min(max(random_p, args.snr_start), args.snr_stop)

            data_val, label_val = self.data_generation_eval(random_p, random_snr)
            data = np.append(data, data_val, axis=0)
            label = np.append(label, label_val, axis=0)

        file_path = f"{data_dir}/validate_set.h5"
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('label', data=label)

if __name__ == '__main__':
    args = parser.parse_args()

    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()

    rawdb = Rawdb(args, encoder_dict, encoder_definite, channel_dict)

    rawdb.build_rawdb("./data")