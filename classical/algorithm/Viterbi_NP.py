import numpy as np
import sys
import os
import copy
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import find_index
sys.path.pop()

## Decoder: Viterbi_NP decoder
class Viterbi_NP(object):
    def __init__(self, params:Params, channel_dict, ini_metric):
        self.params = params
        self.channel_dict = channel_dict
        self.ini_metric = ini_metric
        self.num_state = self.channel_dict['num_state']
    
    def dec(self, r, pred_coef):
        length = r.shape[1]
        dec = np.empty((1, 0))
        params = self.params
        metric_cur = self.ini_metric
        hist_init = {
            'state_path' : np.empty((self.num_state, 0)),
            'pred' : np.zeros((self.num_state, 1)),
            'r' : np.empty((1, 0))
        }
        hist_cur = hist_init
        
        for pos in range(0, length - params.overlap_length, params.eval_length):
            r_truncation = r[:, pos:pos+params.eval_length+params.overlap_length]
            dec_tmp, metric_next, hist_next = self.npml_dec(r_truncation, metric_cur, hist_cur, pred_coef)
            
            hist_cur = hist_next
            metric_cur = metric_next
            dec = np.append(dec, dec_tmp, axis=1)
        return dec
    
    def npml_dec(self, r_truncation, metric_cur, hist_cur, pred_coef):
        params = self.params
        r_len = r_truncation.shape[1]
        metric_cur_trun = metric_cur
        hist_cur_trun = hist_cur
        path_survivor = np.empty((self.num_state, 0))
        
        for idx in range(r_len):
            
            state_path, state_metric = self.metric(r_truncation[:, idx], metric_cur_trun, 
                                                   hist_cur_trun['pred'])
            
            # compute the memory decision with prediction
            
            hist_trun = self.metric_hist(r_truncation[:, idx:idx+1], hist_cur_trun, pred_coef, 
                                              state_path)
            
            hist_cur_trun = hist_trun
            path_survivor = np.append(path_survivor, state_path, axis=1)
            metric_cur_trun = state_metric
            if idx == params.eval_length-1:
                state_metric_next = state_metric
                hist_next_blk = copy.deepcopy(hist_trun)
        
        state_min = np.argmin(state_metric, axis=0)[0]
        path = self.path_convert(path_survivor)
        dec_word, _ = self.path_to_word(path, state_min)
        return dec_word[:, :params.eval_length], state_metric_next, hist_next_blk
    
    def metric_hist(self, r_trun, hist_cur_trun, pred_coef, state_path):
        '''
        update the history of path (dec, pred, r)
        dec the new bit from path
        compute the prediction for the next step
        '''
        params = self.params
        hist_cur_trun['state_path'] = np.append(hist_cur_trun['state_path'], 
                                                state_path, axis=1)
        hist_cur_trun['r'] = np.append(hist_cur_trun['r'], r_trun, axis=1)
        
        if hist_cur_trun['r'].shape[1] > params.noise_predictor_nums:
            hist_cur_trun['r'] = hist_cur_trun['r'][:, -params.noise_predictor_nums:]
            hist_cur_trun['state_path'] = (hist_cur_trun['state_path']
                                           [:, -params.noise_predictor_nums:])
        
        path_survivor_all = hist_cur_trun['state_path']
        path_all = self.path_convert(path_survivor_all)
        dec_out = np.zeros((self.num_state, 
                            min(hist_cur_trun['r'].shape[1], params.noise_predictor_nums)))
        for state in range(self.num_state):
            _, dec_out[state, :] = self.path_to_word(path_all, state)
        
        dec_out_len = dec_out.shape[1]
        if dec_out_len < params.noise_predictor_nums:
            dec_out = np.append(np.zeros((self.num_state, params.noise_predictor_nums-dec_out_len)),
                                dec_out, axis=1)
            r = np.append(np.zeros((1, params.noise_predictor_nums-dec_out_len)), 
                          hist_cur_trun['r'], axis=1)
        else:
            r = hist_cur_trun['r']
        
        for state in range(self.num_state):
            hist_cur_trun['pred'][state, 0] = np.sum(np.multiply((r - dec_out[state, :]), 
                                                                 np.flip(pred_coef)))
        
        return hist_cur_trun
    
    def metric(self, r, metric_last, dec_hist_pred):
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
                state_in = self.channel_dict['state_machine'][set_in[i], 0]
                metric_tmp[i, :] = (metric_last[state_in, :][0] + 
                                    self.euclidean_distance(r-dec_hist_pred[state_in, 0], 
                                                            self.channel_dict['in_out'][set_in[i], 1]))
            metric_survivor[state, :] = metric_tmp.min()
            # if we find equal minimum branch metric, we choose the upper path
            path_survivor[state, :] = (
                self.channel_dict['state_machine'][set_in[np.where(metric_tmp==metric_tmp.min())[0][0]], 0])
        return path_survivor, metric_survivor
    
    def path_convert(self, path_survivor):
        '''
        Input: (num_state, length) array
        Output: (num_state, length) array
        Mapping: Viterbi decoder for a truncation part
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
        Output: (1, length-1) array
        Mapping: connection between two states determines one word
        '''
        
        length = path.shape[1]
        word = np.zeros((1, length))
        output = np.zeros((1, length))
        for i in range(length-1):
            idx = find_index(self.channel_dict['state_machine'], path[state, i : i+2])
            word[:, i] = self.channel_dict['in_out'][idx, 0]
            output[:, i] = self.channel_dict['in_out'][idx, 1]
        
        idx = find_index(self.channel_dict['state_machine'], np.array([path[state, -1], state]))
        word[:, -1] = self.channel_dict['in_out'][idx, 0]
        output[:, -1] = self.channel_dict['in_out'][idx, 1]
        
        return word, output
    
    def euclidean_distance(self, x, y):
        return np.sum((x - y) ** 2)