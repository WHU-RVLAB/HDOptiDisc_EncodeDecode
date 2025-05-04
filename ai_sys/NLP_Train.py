import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import sys
import datetime
np.set_printoptions(threshold=sys.maxsize)

from NLP.BaseModel import BaseModel
from NLP.RNN import RNN
from NLP.RNNScratch import RNNScratch
from NLP.Transformer import Transformer
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Model_Dataset import PthDataset
sys.path.pop()

def main():
    global params
    params = Params()

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # data loader
    train_dataset = PthDataset(file_path='../data/train_set.pth', params=params, model_type = "NLP")
    test_dataset = PthDataset(file_path='../data/test_set.pth', params=params, model_type = "NLP")
    val_dataset = PthDataset(file_path='../data/validate_set.pth', params=params, model_type = "NLP")

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
    
    # model dir
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
        
    model_path = f"{params.model_dir}/{model_file}"
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size_train, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size_test, shuffle=False, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size_val, shuffle=False, drop_last=True, num_workers=4)
    
    # criterion and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=params.learning_rate, 
                                eps=1e-08, 
                                weight_decay=params.weight_decay)

    # output dir 
    dir_name = '../output/output_' + datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S') + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    result_path = dir_name + params.result_file
    
    try:
        result = open(result_path, 'w+')
        print(f"File {result_path} opened successfully.")
    except IOError as e:
        print(f"Error opening file {result_path}: {e}")

    # train and validation
    for epoch in range(params.num_epoch):
        
        # train and validate
        train_loss = train(train_loader, model, optimizer, epoch, device)
        test_loss, ber = validate(test_loader, val_loader, model, epoch, device)
        
        result.write('epoch %d \n' % epoch)
        result.write('Train loss:'+ str(train_loss)+'\n')
        result.write('Test loss:'+ str(test_loss)+'\n')
        if (epoch >= params.eval_start and epoch % params.eval_freq == 0):
            result.write('-----evaluation ber:'+str(ber)+'\n')
        else:
            result.write('-----:no evaluation'+'\n')
        result.write('\n')
        result.flush()
        
        torch.save({
            'epoch': epoch+1,
            'arch': params.model_arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path, pickle_protocol=4)
    result.close()
    
def train(train_loader, model:BaseModel, optimizer, epoch, device):
    # switch to train mode
    model.train()
    
    train_loss = 0
    bt_cnt = 0
    
    if params.model_arch == "rnn" or params.model_arch == "rnn_scratch":
        hidden_dim = params.rnn_hidden_size
        rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
        num_layers = rnn_hidden_size_factor*params.rnn_layer
    elif params.model_arch == "transformer":
        hidden_dim = params.transformer_hidden_size
        num_layers = params.transformer_decoder_layers
    
    for datas, labels in train_loader:
        batch_size, seq_length, _ = datas.shape
        datas, labels = datas.to(device), labels.to(device)
        init_hidden = torch.zeros(num_layers, params.batch_size_train, hidden_dim, device=device).detach()
        cur_hidden = init_hidden
        for pos in range(0, seq_length - params.post_overlap_length, params.eval_length):            
            datas_truncation = datas[:, pos:pos+params.eval_length+params.post_overlap_length, :]
            labels_truncation = labels[:, pos:pos+params.eval_length+params.post_overlap_length]
            output, nxt_hidden = model(datas_truncation, cur_hidden)
            loss = loss_func(output, labels_truncation, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
            bt_cnt += 1 
            if params.nlp_transmis_hidden:         
                cur_hidden = nxt_hidden.detach()
    avg_loss = train_loss / bt_cnt

    # print
    if (epoch % params.print_freq_ep == 0):
        print('Train Epoch: {} Avg Loss: {:.6f}'.format(epoch+1, avg_loss))
    
    return avg_loss
            

def validate(test_loader, val_loader, model:BaseModel, epoch, device):
    # switch to evaluate mode
    model.eval()
    
    if params.model_arch == "rnn" or params.model_arch == "rnn_scratch":
        hidden_dim = params.rnn_hidden_size
        rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
        num_layers = rnn_hidden_size_factor*params.rnn_layer
    elif params.model_arch == "transformer":
        hidden_dim = params.transformer_hidden_size
        num_layers = params.transformer_decoder_layers
    
    # network
    with torch.no_grad():
        test_loss = 0
        bt_cnt = 0
        for datas, labels in test_loader:
            datas, labels = datas.to(device), labels.to(device)
            init_hidden = torch.zeros(num_layers, params.batch_size_test, hidden_dim, device=device)
            output, _ = model(datas, init_hidden)
            loss = loss_func(output, labels, device)

            test_loss += loss.item()
            bt_cnt += 1
        avg_loss = test_loss / bt_cnt
    
    if epoch % params.print_freq_ep == 0:
        print('Test Epoch: {} Avg Loss: {:.6f}'.format(epoch+1, avg_loss))

    # evaluation
    ber = 1.0
    if (epoch >= params.eval_start) & (epoch % params.eval_freq == 0):
        decodeword = np.empty((1, 0))
        label_val = np.empty((1, 0))
        for datas, labels in val_loader:
            init_hidden = torch.zeros(num_layers, params.batch_size_val, hidden_dim, device=device)
            datas = datas.to(device)
            dec, _ = model.decode(datas, init_hidden)
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