import os
import argparse
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import math
import sys
import h5py
import datetime
np.set_printoptions(threshold=sys.maxsize)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Utils import codeword_threshold
from Dataset import H5Dataset
sys.path.pop()

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

parser.add_argument('-eval_length', type=int, default=10)
parser.add_argument('-overlap_length', type=int, default=20)

parser.add_argument('-input_size', type=int, default=5)
parser.add_argument('-rnn_input_size', type=int, default=5)
parser.add_argument('-rnn_hidden_size', type=int, default=50)
parser.add_argument('-output_size', type=int, default=1)
parser.add_argument('-rnn_layer', type=int, default=4)
parser.add_argument('-rnn_dropout_ratio', type=float, default=0)

parser.add_argument('-batch_size_train', type=int, default=30)
parser.add_argument('-batch_size_test', type=int, default=600)
parser.add_argument('-batch_size_val', type=int, default=600)

def main():

    global args
    args = parser.parse_known_args()[0]

    # data loader
    train_dataset = H5Dataset(h5_file='./data/train_set.h5')
    test_dataset = H5Dataset(h5_file='./data/test_set.h5')
    val_dataset = H5Dataset(h5_file='./data/validate_set.h5')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=0)
    
    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # model
    model = Network(args, device).to(device)
    
    # criterion and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=args.learning_rate, 
                                 eps=1e-08, 
                                 weight_decay=args.weight_decay)
    
    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.epoch_start = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # output dir 
    dir_name = './output/output_' + datetime.datetime.strftime(datetime.datetime.now(), 
                                                        '%Y_%m_%d_%H_%M_%S') + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_path = dir_name + args.result
    result = open(result_path, 'w+')

    # train and validation
    for epoch in range(args.epoch_start, args.num_epoch):
        
        # train and validate
        train_loss = train(train_loader, model, optimizer, epoch, device)
        valid_loss, ber_acgn_1, ber_acgn_2 = validate(test_loader, val_loader, model, epoch, device)
        
        result.write('epoch %d \n' % epoch)
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
    
class Network(nn.Module):
    def __init__(self, args, device):
        super(Network, self).__init__()
        
        self.args = args
        self.device = device
        self.time_step = (args.eval_length + args.overlap_length)
        self.fc_length = args.eval_length + args.overlap_length
        self.dec_input = torch.nn.Linear(args.input_size, 
                                         args.rnn_input_size)
        self.dec_rnn = torch.nn.GRU(args.rnn_input_size, 
                                    args.rnn_hidden_size, 
                                    args.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=args.rnn_dropout_ratio, 
                                    bidirectional=True)
        
        self.dec_output = torch.nn.Linear(2*args.rnn_hidden_size, args.output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        dec = torch.zeros(batch_size, self.fc_length, self.args.output_size).to(self.device)
        
        x = self.dec_input(x)
        y, _  = self.dec_rnn(x)
        y_dec = y[:, :self.time_step, :]

        dec = torch.sigmoid(self.dec_output(y_dec))
        
        return torch.squeeze(dec, 2)
    

def train(train_loader, model, optimizer, epoch, device):
    # switch to train mode
    model.train()
    
    train_loss = 0
    for batch_idx in range(args.num_batch):
        for datas, labels in train_loader:
            # network
            optimizer.zero_grad()
            output = model(datas)
            loss = loss_func(output, labels)

            # compute gradient and do gradient step
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # print
        if (epoch % args.print_freq_ep == 0 and 
            (batch_idx+1) % args.num_batch == 0):
            avg_loss = train_loss / args.num_batch
            print('Train Epoch: {} (Loss: {:.6f}, Avg Loss: {:.6f}'
                  .format(epoch+1, train_loss, avg_loss))
    
    return loss.item()
            

def validate(test_loader, val_loader, model, epoch, device):
    # switch to evaluate mode
    model.eval()
        
    # network
    with torch.no_grad():
        for datas, labels in test_loader:
            output = model(datas)
            test_loss = loss_func(output, labels)
    
    if epoch % args.print_freq_ep == 0:
        print('Tset Epoch: {} - Loss: {:.6f}'.format(epoch+1, test_loss.item()))
    
    # evaluation
    if (epoch >= args.eval_start) & (epoch % args.eval_freq == 0):
        decodeword = np.empty((1, 0))
        label_val = np.empty((1, 0))
        for datas, labels in val_loader:
            dec = evaluation(datas, model, device).cpu().numpy()
            decodeword = np.append(decodeword, dec, axis=1)
            labels = labels.reshape(-1)
            label_val = np.append(label_val, labels, axis=1)
        ber = (np.sum(np.abs(decodeword - label_val))/label_val.shape[0])
        print('Validation Epoch: {} - ber: {}'.format(epoch+1, ber))
    
    return test_loss.item(), ber
        
def evaluation(data_eval, model, device):
    dec = torch.zeros((1, 0)).float().to(device)
    for idx in range(data_eval.shape[0]):
        truncation_in = torch.from_numpy(data_eval[idx:idx + 1, : , :]).float().to(device)
        with torch.no_grad():
            dec_block = codeword_threshold(model(truncation_in)[:, :args.eval_length])
        # concatenate the decoding codeword
        dec = torch.cat((dec, dec_block), 1)
        
    return dec

def loss_func(output, label):
    return F.binary_cross_entropy(output, label).cpu()
        
if __name__ == '__main__':
    main()