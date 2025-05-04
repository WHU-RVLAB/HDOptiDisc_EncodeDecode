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
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Params import Params
from lib.Utils import sliding_shape
sys.path.pop()

class PthDataset(Dataset):
    def __init__(self, file_path, params:Params, model_type="NLP"):
        data = torch.load(file_path, weights_only=False)
        if model_type == "Classifier":
            data_np = sliding_shape(data['data'][:10000, :], params.classifier_input_size)
            label_np = data['label']
        elif model_type == "NLP":
            data_np = sliding_shape(data['data'][:10000, :], params.nlp_input_size)
            label_np = data['label']
        self.data = torch.from_numpy(data_np).float()
        self.label = torch.from_numpy(label_np).float()

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
    
    def data_generation(self, prob):
        '''
        training/testing data(without sliding window) and label
        output: numpy array 
        '''
        params = self.params
        num_snr = int((params.model_snr_stop-params.model_snr_start)/params.model_snr_step+1)
        snr_size = params.model_snr_size
        bt_size = num_snr*snr_size
        block_length = params.block_length
        
        data, label = (np.zeros((bt_size, block_length)), np.zeros((bt_size, block_length)))
        
        for snr_idx in np.arange(0, num_snr):
            snr = params.model_snr_start+snr_idx*params.model_snr_step
            for signal_idx in np.arange(0, snr_size):
                info = np.random.choice(np.arange(0, 2), size = (1, int(params.block_length*self.code_rate)), p=[1-prob, prob])
                
                codeword = self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info))
                signal_upsample_ideal, signal_upsample_jittered, rf_signal_ideal, rf_signal = self.disk_read_channel.RF_signal_jitter(codeword)
                if params.jitteron:
                    rf_signal_input = rf_signal
                else:
                    rf_signal_input = rf_signal_ideal
                equalizer_input = self.disk_read_channel.awgn(rf_signal_input, snr)   
                label[snr_idx*snr_size + signal_idx:snr_idx*snr_size + signal_idx + 1, :] = codeword
                data[snr_idx*snr_size + signal_idx:snr_idx*snr_size + signal_idx + 1, :]  = equalizer_input
        
        print("generate training/testing data(without sliding window) and label")
        
        return data, label
    
    def data_generation_eval(self, prob, snr):
        '''
        evaluation data (without sliding window) and label
        output: numpy array data_eval, numpy array label_eval
        '''
        params = self.params
        snr_size = params.model_snr_size
        bt_size = snr_size
        block_length = params.block_length
        
        data, label = (np.zeros((bt_size, block_length)), np.zeros((bt_size, block_length)))
        
        for signal_idx in np.arange(0, snr_size):
            info = np.random.choice(np.arange(0, 2), size = (1, int(params.block_length*self.code_rate)), p=[1-prob, prob])
            
            codeword = self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info))
            signal_upsample_ideal, signal_upsample_jittered, rf_signal_ideal, rf_signal = self.disk_read_channel.RF_signal_jitter(codeword)
            if params.jitteron:
                rf_signal_input = rf_signal
            else:
                rf_signal_input = rf_signal_ideal
            equalizer_input = self.disk_read_channel.awgn(rf_signal_input, snr)   
            label[signal_idx:signal_idx + 1, :] = codeword
            data[signal_idx:signal_idx + 1, :]  = equalizer_input
        
        print("generate evaluation data (without sliding window) and label")
        
        return data, label
    
    def build_rawdb(self, data_dir):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        params =  self.params
        block_length = params.block_length

        data = np.empty((0, block_length))
        label = np.empty((0, block_length))
        for _ in range(params.train_num_probs):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_train, label_train = self.data_generation(random_p)
            data = np.append(data, data_train, axis=0)
            label = np.append(label, label_train, axis=0)

        file_path = f"{data_dir}/train_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path, pickle_protocol=4)
        print("generate training dataset\n")

        data = np.empty((0, block_length))
        label = np.empty((0, block_length))
        for _ in range(params.test_num_probs):

            miu = (0.1 + 0.9)/2
            sigma = (0.9 - miu)/2
            random_p = np.random.normal(miu, sigma)
            random_p = min(max(random_p, 0), 1)

            data_test, label_test = self.data_generation(random_p)
            data = np.append(data, data_test, axis=0)
            label = np.append(label, label_test, axis=0)

        file_path = f"{data_dir}/test_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path, pickle_protocol=4)
        print("generate testing dataset\n")

        data = np.empty((0, block_length))
        label = np.empty((0, block_length))
        for _ in range(params.val_num_probs):

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

        file_path = f"{data_dir}/validate_set.pth"
        torch.save({
            'data': data,
            'label': label
        }, file_path, pickle_protocol=4)
        print("generate validate dataset\n")

if __name__ == '__main__':
    params = Params()

    # constant and input paras
    encoder_dict, encoder_definite = RLL_state_machine()
    channel_dict = Target_channel_state_machine()
    if params.signal_norm:
        channel_dict['in_out'][:, 1] /= sum(params.PR_coefs)

    rawdb = Rawdb(params, encoder_dict, encoder_definite, channel_dict)

    rawdb.build_rawdb("../data")