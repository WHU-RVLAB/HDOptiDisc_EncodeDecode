import numpy as np
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import find_index
sys.path.pop()

## Detector: Viterbi detector
class Viterbi(object):
    def __init__(self, params:Params, channel_dict, ini_metric):
        self.params = params
        self.channel_dict = channel_dict
        self.ini_metric = ini_metric
        self.num_state = self.channel_dict['num_state']
    
    def vit_dec(self, r_truncation, ini_metric):
        
        r_len = r_truncation.shape[1]
        ini_metric_trun = ini_metric
        path_survivor = np.zeros((self.num_state, r_len))
        state_metric_trun = np.zeros((self.num_state, self.params.eval_length))
        
        for idx in range(r_len):
            state_path, state_metric = self.metric(r_truncation[:, idx], 
                                                   ini_metric_trun)
            
            ini_metric_trun = state_metric
            path_survivor[:, idx:idx+1] = state_path
            if idx == self.params.eval_length-1:
                state_metric_next = state_metric
        
        state_min = np.argmin(state_metric, axis=0)[0]
        path = self.path_convert(path_survivor)
        dec_word = self.path_to_word(path, state_min)
        
        return dec_word[:, :self.params.eval_length], state_metric_next
        
    def metric(self, r, metric_last):
        '''
        Input: branch metrics at one time step
        Output: branch metric and survivor metric for the next step 
        Mapping: choose the shorest path between adjacent states
        '''
        
        path_survivor, metric_survivor = (np.zeros((self.num_state, 1)), 
                                          np.zeros((self.num_state, 1)))
        
        for state in range(self.num_state):
            set_in = np.where(self.channel_dict['state_machine'][:, 1]==state)[0]
            metric_tmp = np.zeros((set_in.shape[0], 1))
            for i in range(set_in.shape[0]):
                metric_tmp[i, :] = (metric_last[self.channel_dict['state_machine'][set_in[i], 0], :][0] + 
                                    self.euclidean_distance(r, self.channel_dict['in_out'][set_in[i], 1]))
            metric_survivor[state, :] = metric_tmp.min()
            # if we find equal minimum branch metric, we choose the upper path
            path_survivor[state, :] = (
                self.channel_dict['state_machine'][set_in[np.where(metric_tmp==metric_tmp.min())[0][0]], 0])
        return path_survivor, metric_survivor
                
    
    def path_convert(self, path_survivor):
        '''
        Input: (num_state, length) array
        Output: (num_state, length) array
        Mapping: Viterbi detector for a truncation part
        '''
        
        path_truncation = np.zeros(path_survivor.shape)
        path_truncation[:, -1:] = path_survivor[:, -1:]
        for state in range(self.num_state):
            for i in range(path_survivor.shape[1]-2, -1, -1):
                path_truncation[state, i] = int(path_survivor[
                    int(path_truncation[state, i+1]), i])
        
        return path_truncation
    
    def path_to_word(self, path, state):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: connection between two states determines one word
        '''
        
        length = path.shape[1]
        word = np.zeros((1, length))
        for i in range(length-1):
            idx = find_index(self.channel_dict['state_machine'], path[state, i : i+2])
            word[:, i] = self.channel_dict['in_out'][idx, 0]
        
        idx = find_index(self.channel_dict['state_machine'], np.array([path[state, -1], state]))
        word[:, -1] = self.channel_dict['in_out'][idx, 0]
        return word
    
    def euclidean_distance(self, x, y):
        return np.sum((x - y) ** 2)