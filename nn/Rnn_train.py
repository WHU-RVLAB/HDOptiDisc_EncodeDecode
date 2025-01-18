import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import sys
import datetime
np.set_printoptions(threshold=sys.maxsize)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Utils import evaluation
from Dataset import PthDataset
sys.path.pop()

parser = argparse.ArgumentParser()

parser.add_argument('-learning_rate', type = float, default=0.001)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-weight_decay', type=float, default=0.0001)

parser.add_argument('-num_epoch', type=int, default=400)
parser.add_argument('-epoch_start', type=int, default=0)
parser.add_argument('-eval_freq', type=int, default=5)
parser.add_argument('-eval_start', type=int, default=200)
parser.add_argument('-print_freq_ep', type=int, default=5)

parser.add_argument('-model_dir', default="../model/", type=str, metavar='PATH', help='path to latest model')
parser.add_argument('-result', type=str, default='result.txt')
parser.add_argument('-model_name', type=str, default='model.pth.tar')

parser.add_argument('-eval_length', type=int, default=30)
parser.add_argument('-overlap_length', type=int, default=30)

parser.add_argument('-input_size', type=int, default=5)
parser.add_argument('-rnn_input_size', type=int, default=5)
parser.add_argument('-rnn_hidden_size', type=int, default=50)
parser.add_argument('-output_size', type=int, default=1)
parser.add_argument('-rnn_layer', type=int, default=10)
parser.add_argument('-rnn_dropout_ratio', type=float, default=0)

parser.add_argument('-batch_size_train', type=int, default=300)
parser.add_argument('-batch_size_test', type=int, default=600)
parser.add_argument('-batch_size_val', type=int, default=600)

def main():

    global args
    args = parser.parse_known_args()[0]

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # data loader
    train_dataset = PthDataset(file_path='../data/train_set.pth')
    test_dataset = PthDataset(file_path='../data/test_set.pth')
    val_dataset = PthDataset(file_path='../data/validate_set.pth')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)

    # model
    model = RNN(args, device).to(device)
    
    # criterion and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=args.learning_rate, 
                                 eps=1e-08, 
                                 weight_decay=args.weight_decay)
    
    # model dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = f"{args.model_dir}/{args.model_name}"

    # output dir 
    dir_name = '../output/output_' + datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S') + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_path = dir_name + args.result
    result = open(result_path, 'w+')

    # train and validation
    for epoch in range(args.epoch_start, args.num_epoch):
        
        # train and validate
        train_loss = train(train_loader, model, optimizer, epoch, device)
        valid_loss, ber = validate(test_loader, val_loader, model, epoch, device)
        
        result.write('epoch %d \n' % epoch)
        result.write('Train loss:'+ str(train_loss)+'\n')
        result.write('Validation loss:'+ str(valid_loss)+'\n')
        if (epoch >= args.eval_start and epoch % args.eval_freq == 0):
            result.write('-----evaluation ber:'+str(ber)+'\n')
        else:
            result.write('-----:no evaluation'+'\n')
        result.write('\n')
        
        torch.save({
            'epoch': epoch+1,
            'arch': 'rnn',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)
    
class RNN(nn.Module):
    def __init__(self, args, device):
        super(RNN, self).__init__()
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
    bt_cnt = 0
    for datas, labels in train_loader:
        datas, labels = datas.to(device), labels.to(device)
        # network
        optimizer.zero_grad()
        output = model(datas)
        loss = loss_func(output, labels, device)

        # compute gradient and do gradient step
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        bt_cnt += 1
    avg_loss = train_loss / bt_cnt

    # print
    if (epoch % args.print_freq_ep == 0):
        print('Train Epoch: {} Avg Loss: {:.6f}'.format(epoch+1, avg_loss))
    
    return avg_loss
            

def validate(test_loader, val_loader, model, epoch, device):
    # switch to evaluate mode
    model.eval()
        
    # network
    with torch.no_grad():
        test_loss = 0
        bt_cnt = 0
        for datas, labels in test_loader:
            datas, labels = datas.to(device), labels.to(device)
            output = model(datas)
            loss = loss_func(output, labels, device)
            test_loss += loss.item()
            bt_cnt += 1
        avg_loss = test_loss / bt_cnt
    
    if epoch % args.print_freq_ep == 0:
        print('Test Epoch: {} Avg Loss: {:.6f}'.format(epoch+1, avg_loss))
    
    # evaluation
    ber = 1.0
    if (epoch >= args.eval_start) & (epoch % args.eval_freq == 0):
        decodeword = np.empty((1, 0))
        label_val = np.empty((1, 0))
        for datas, labels in val_loader:
            datas = datas.to(device)
            dec = evaluation(args.eval_length, datas, model, device)
            decodeword = np.append(decodeword, dec, axis=1)
            labels = labels.numpy().reshape(1, -1)
            label_val = np.append(label_val, labels, axis=1)
        ber = (np.sum(np.abs(decodeword - label_val))/label_val.shape[1])
        print('Validation Epoch: {} - ber: {}'.format(epoch+1, ber))
    
    return avg_loss, ber

def loss_func(output, label, device):
    return F.binary_cross_entropy(output, label).to(device)
        
if __name__ == '__main__':
    main()