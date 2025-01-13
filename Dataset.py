import os
import argparse
import torch
import numpy as np
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

from lib.Const import RLL_state_machine, Target_channel_state_machine, Target_channel_dummy_bits
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel
from lib.Channel_Modulator import RLL_Modulator
from lib.Channel_Converter import NRZI_Converter
from lib.Disk_Read_Channel import Disk_Read_Channel

parser = argparse.ArgumentParser()

parser.add_argument('-learning_rate', type = float, default=0.001)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-num_epoch', type=int, default=400)
parser.add_argument('-epoch_start', type=int, default=0)
parser.add_argument('-num_batch', type=int, default=20)
parser.add_argument('-weight_decay', type=float, default=0.0001)
parser.add_argument('-eval_freq', type=int, default=5)
parser.add_argument('-eval_start', type=int, default=200)
parser.add_argument('-print_freq_ep', type=int, default=5)

parser.add_argument('-result', type=str, default='result.txt')
parser.add_argument('-checkpoint', type=str, default='checkpoint.pth.tar')
parser.add_argument('-resume', default=None, type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-prob_start', type=float, default=0.1)
parser.add_argument('-prob_up', type=float, default=0.01)
parser.add_argument('-prob_step_ep', type=int, default=5)

parser.add_argument('-eval_info_length', type=int, default=1000000)
parser.add_argument('-dummy_length_start', type=int, default=5)
parser.add_argument('-dummy_length_end', type=int, default=5)
parser.add_argument('-eval_length', type=int, default=10)
parser.add_argument('-overlap_length', type=int, default=20)

parser.add_argument('-batch_size_snr_train_1', type=int, default=30)
parser.add_argument('-batch_size_snr_train_2', type=int, default=30)
parser.add_argument('-batch_size_snr_validate_1', type=int, default=600)
parser.add_argument('-batch_size_snr_validate_2', type=int, default=600)
parser.add_argument('-snr_start', type=float, default=8.5)
parser.add_argument('-snr_stop', type=float, default=10.5)
parser.add_argument('-snr_step', type=float, default=0.5)

parser.add_argument('-input_size', type=int, default=5)
parser.add_argument('-rnn_input_size', type=int, default=5)
parser.add_argument('-rnn_hidden_size', type=int, default=50)
parser.add_argument('-output_size', type=int, default=1)
parser.add_argument('-rnn_layer', type=int, default=4)
parser.add_argument('-rnn_dropout_ratio', type=float, default=0)

parser.add_argument('-scaling_para', type=float, default=0.25)
parser.add_argument('-PW50_1', type=float, default=2.54)
parser.add_argument('-PW50_2', type=float, default=2.88)
parser.add_argument('-T', type=float, default=1)
parser.add_argument('-tap_lor_num', type=int, default=41)
parser.add_argument('-tap_isi_num', type=int, default=21)
parser.add_argument('-tap_pre_num', type=int, default=4)

global args
args = parser.parse_args()

## Dataset: generate dataset for neural network
class Dataset(object):
    def __init__(self, args, device, encoder_dict, encoder_definite, channel_dict):
        self.args = args
        self.device = device
        
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
        output: float torch tensor (device)
        '''
        
        bt_size_1 = int(((args.snr_stop-args.snr_start)/
                         args.snr_step+1)*bt_size_snr_1)
        bt_size_2 = int(((args.snr_stop-args.snr_start)/
                         args.snr_step+1)*bt_size_snr_2)
        
        block_length = args.eval_length + args.overlap_length
        info_length = math.ceil(block_length/self.num_out_sym)*self.num_input_sym_enc
        
        info_1 = np.random.choice(np.arange(0, 2), size = (bt_size_snr_1, info_length), 
                                  p=[1-prob, prob])
        
        info_2 = np.random.choice(np.arange(0, 2), size = (bt_size_snr_2, info_length), 
                                  p=[1-prob, prob])
        
        data_bt_1, label_bt_1 = (np.zeros((bt_size_1, block_length)), 
                                 np.zeros((bt_size_1, block_length)))
        data_bt_2, label_bt_2 = (np.zeros((bt_size_2, block_length)), 
                                 np.zeros((bt_size_2, block_length)))
                
        for i in range(bt_size_snr_1):
            codeword = (self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info_1[i : i+1, :]))[:, :block_length])
            
            for idx in np.arange(0, (args.snr_stop-args.snr_start)/args.snr_step+1):
                label_bt_1[int(idx*bt_size_snr_1+i) : int(idx*bt_size_snr_1+i+1), :] = codeword
                
                rf_signal = self.disk_read_channel.RF_signal(codeword)
                equalizer_input = self.disk_read_channel.awgn(rf_signal, args.snr_start+idx*args.snr_step)
                
                data_bt_1[int(idx*bt_size_snr_1+i) : int(idx*bt_size_snr_1+i+1), :] = equalizer_input
        
        for i in range(bt_size_snr_2):
            codeword = (self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info_2[i : i+1, :]))[:, :block_length])
            
            for idx in np.arange(0, (args.snr_stop-args.snr_start)/args.snr_step+1):
                label_bt_2[int(idx*bt_size_snr_2+i) : int(idx*bt_size_snr_2+i+1), :] = codeword
                
                rf_signal = self.disk_read_channel.RF_signal(codeword)
                equalizer_input = self.disk_read_channel.awgn(rf_signal, args.snr_start+idx*args.snr_step)
                
                data_bt_2[int(idx*bt_size_snr_2+i) : int(idx*bt_size_snr_2+i+1), :] = equalizer_input
        
        data_bt = np.append(data_bt_1, data_bt_2, axis=0)
        label_bt = np.append(label_bt_1, label_bt_2, axis=0)
        
        data_bt = self.sliding_shape(torch.from_numpy(data_bt).float().to(self.device))
        label_bt = (torch.from_numpy(label_bt).float()).to(self.device)
        
        return data_bt, label_bt
    
    def data_generation_eval(self, snr):
        '''
        evaluation data(without sliding window) and label
        output: float torch tensor data_eval, numpy array label_eval
        '''
        info = np.random.randint(2, size = (1, args.eval_info_length))
        codeword = self.NRZI_converter.forward_coding(self.RLL_modulator.forward_coding(info))
        
        rf_signal = self.disk_read_channel.RF_signal(codeword)
        equalizer_input = self.disk_read_channel.awgn(rf_signal, snr)

        equalizer_input = torch.from_numpy(equalizer_input).float().to(self.device)
        codeword = torch.from_numpy(codeword).float().to(self.device)
        
        return equalizer_input, codeword
        
    def sliding_shape(self, x):
        '''
        Input: (1, length) torch tensor
        Output: (input_size, length) torch tensor
        Mapping: sliding window for each time step
        '''
        
        batch_size, time_step = x.shape
        zero_padding_len = args.input_size - 1
        x = torch.cat(((torch.zeros((batch_size, zero_padding_len))).to(self.device), x), 1)
        y = torch.zeros(batch_size, time_step, args.input_size)
        for bt in range(batch_size):
            for time in range(time_step):
                y[bt, time, :] = x[bt, time:time+args.input_size]
        return y.float().to(self.device)
    
    def build_dataset(self):

        # cuda device
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        if torch.cuda.is_available():
            device = torch.device("cuda")
            
        # write the results
        dir_name = './output_' + datetime.datetime.strftime(datetime.datetime.now(), 
                                                            '%Y-%m-%d_%H:%M:%S') + '/'
        os.mkdir(dir_name)
        result_path = dir_name + args.result
        result = open(result_path, 'w+')
        
        # constant and input paras
        encoder_dict, encoder_definite = RLL_state_machine()
        channel_dict = Target_channel_state_machine()

        data_class = Dataset(args, device, encoder_dict, encoder_definite, channel_dict)
        data_eval_acgn_1, data_eval_acgn_2, label_eval = (data_class.data_generation_eval(8.5))
        
        snr_point = int((args.snr_stop-args.snr_start)/args.snr_step+1)
        
        # model
        model = Network(args, device).to(device)
        
        # criterion and optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=args.learning_rate, 
                                    eps=1e-08, 
                                    weight_decay=args.weight_decay)
        
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.epoch_start = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        
        # train and validation
        
        prob_start_ori = 0.1
        if args.epoch_start > args.eval_start:
            prob_start = 0.5
        else:
            prob_start = prob_start_ori + (args.prob_up * 
                                        (args.epoch_start // args.prob_step_ep))
        
        prob_end = 0.5
        prob_step = int((prob_end - prob_start_ori) / args.prob_up)
        prob_ep_list = list(range(prob_step*args.prob_step_ep, 0, -args.prob_step_ep))
        prob = prob_start
        for epoch in range(args.epoch_start, args.num_epoch):
            
            # increase the probability each 10 epochs
            if epoch in prob_ep_list:
                prob += args.prob_up
            
            # train and validate
            train_loss = train(data_class, prob, model, optimizer, epoch, device)
            valid_loss, ber_acgn_1, ber_acgn_2 = validate(data_class, prob, channel_dict, dummy_dict_start, 
                                                        dummy_dict_end_eval, model, epoch, device)
            
            result.write('epoch %d \n' % epoch)
            result.write('information prob.'+str(prob)+'\n')
            result.write('Train loss:'+ str(train_loss)+'\n')
            result.write('Validation loss:'+ str(valid_loss)+'\n')
            if (epoch >= args.eval_start and epoch % args.eval_freq == 0):
                result.write('-----[acgn] [PW50_1] SNR[dB]:'+str(ber_acgn_1)+'\n')
                result.write('-----[acgn] [PW50_2] SNR[dB]:'+str(ber_acgn_2)+'\n')
            else:
                result.write('-----:no evaluation'+'\n')
            result.write('\n')
            
            torch.save({
                'epoch': epoch+1,
                'arch': 'rnn',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args.checkpoint)