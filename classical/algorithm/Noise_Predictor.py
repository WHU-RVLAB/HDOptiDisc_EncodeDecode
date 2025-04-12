import numpy as np
from numpy import linalg as LA
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
sys.path.pop()

class Noise_Predictor(object):
    
    def __init__(self, params:Params):
        self.params = params
    
    def predictor(self, equalizer_coef, snr):
        params = self.params
        pred_matrix = np.zeros((params.noise_predictor_nums, params.noise_predictor_nums))
        noise_pw = params.bd_scaling_para * 10 ** (- snr * 1.0 / 10)
        for row in range(1, params.noise_predictor_nums+1):
            for col in range(1, params.noise_predictor_nums+1):
                pred_matrix[row-1, col-1] = self.auto_corr(row-col, equalizer_coef, noise_pw)
        pred_const = np.zeros((params.noise_predictor_nums, 1))
        for col in range(1, params.noise_predictor_nums+1):
            pred_const[col-1, 0] = self.auto_corr(col, equalizer_coef, noise_pw)
        pred_coef = np.matmul(LA.inv(pred_matrix), pred_const).transpose()
        
        # MMSE
        mmse_part = 0
        for idx1 in range(1, params.noise_predictor_nums+1):
            for idx2 in range(1, params.noise_predictor_nums+1):
                mmse_part += (pred_coef[0, idx1-1] * pred_coef[0, idx2-1] * 
                              self.auto_corr(idx2-idx1, equalizer_coef, noise_pw))
        
        
        mmse = (self.auto_corr(0, equalizer_coef, noise_pw) - 
                2*np.sum(np.multiply(pred_coef, pred_const.transpose()))
                +mmse_part)

        return pred_coef, mmse
    
    def auto_corr(self, x, equalizer_coef, noise_pw):
        params = self.params
        output = 0
        equalizer_taps_num_side = int((params.equalizer_taps_num - 1) / 2)
        
        for idx1 in range(-equalizer_taps_num_side, equalizer_taps_num_side+1):
            for idx2 in range(-equalizer_taps_num_side, equalizer_taps_num_side+1):
                output += noise_pw * equalizer_coef[0, idx1] * equalizer_coef[0, idx2] * self.delta(idx2 - idx1 + x)
        return output
    
    def delta(self, x):
        if x == 0:
            ans = 1
        else:
            ans = 0
        return ans 