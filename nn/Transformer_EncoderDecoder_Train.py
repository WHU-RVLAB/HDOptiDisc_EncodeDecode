import os

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
from lib.Utils import convert2transformer
from nn.Transformer_EncoderDecoder_Dataset import PthDataset
from lib.Params import Params
from Transformer import Transformer
sys.path.pop()

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(2) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing)
        true_dist.scatter_(2, target.data.long(), self.confidence)
        return self.criterion(x, true_dist)/(x.size(0)*x.size(1))

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
    train_dataset = PthDataset(file_path='../data/transformer_encoderdecoder_train_set.pth')
    test_dataset = PthDataset(file_path='../data/transformer_encoderdecoder_test_set.pth')
    val_dataset = PthDataset(file_path='../data/transformer_encoderdecoder_validate_set.pth')

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size_train, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size_test, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size_val, shuffle=False, num_workers=4)

    # model
    model_file = None
    if params.model_arch == "transformer":
        model = Transformer(params, device).to(device)
        model_file = "transformer.pth.tar"
        
    # criterion and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=params.learning_rate, 
                                 eps=1e-08, 
                                 weight_decay=params.weight_decay)
    
    global loss_func
    loss_func = LabelSmoothing(size=params.transformer_tgt_vocab, smoothing=0.0)
    
    # model dir
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
        
    model_path = f"{params.model_dir}/{model_file}"

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
        }, model_path)
    result.close()
        
def train(train_loader, model:Transformer, optimizer, epoch, device):
    # switch to train mode
    model.train()
    
    train_loss = 0
    bt_cnt = 0
    for datas, labels in train_loader:
        src, target_input, target_pred, target_mask = convert2transformer(datas, labels, params.transformer_nhead, device)
        
        # network
        optimizer.zero_grad()
        
        output = model.forward(src, target_input, target_mask)
        loss = loss_func(model.generator(output), target_pred)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        bt_cnt += 1
    avg_loss = train_loss / bt_cnt

    # print
    if (epoch % params.print_freq_ep == 0):
        print('Train Epoch: {} Avg Loss: {:.6f}'.format(epoch+1, avg_loss))
    
    return avg_loss
            

def validate(test_loader, val_loader, model:Transformer, epoch, device):
    # switch to evaluate mode, but looks like a bug in torch
    # model.eval()
        
    # network
    with torch.no_grad():
        test_loss = 0
        bt_cnt = 0
        for datas, labels in test_loader:
            src, target_input, target_pred, target_mask = convert2transformer(datas, labels, params.transformer_nhead, device)
            
            output = model.forward(src, target_input, target_mask)
            loss = loss_func(model.generator(output), target_pred)
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
                src, target_input, target_pred, target_mask = convert2transformer(datas, labels, params.transformer_nhead, device)

                dec = model.greedy_decode(src, target_pred, max_len=params.transformer_decode_max_len, start_symbol=0, end_symbol=1)
                dec = dec.numpy()[:, :params.eval_length].reshape(1, -1)
                decodeword = np.append(decodeword, dec, axis=1)
                labels = labels.numpy()[:, :params.eval_length].reshape(1, -1)
                label_val = np.append(label_val, labels, axis=1)
            ber = (np.sum(np.abs(decodeword - label_val))/label_val.shape[1])
            print('Validation Epoch: {} - ber: {}'.format(epoch+1, ber))
    
    return avg_loss, ber
        
if __name__ == '__main__':
    main()