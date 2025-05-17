import sys
import os
import numpy as np

import ops.Linear
import ops.RNN

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from BaseModel import BaseModel
sys.path.pop()

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
import ops
sys.path.pop()

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))))
from lib.Params import Params
sys.path.pop()

class RNN(BaseModel):
    def __init__(self, params:Params):
        super(RNN, self).__init__(params)
        weights_dir = 'ai_sys/deploy/weights'
        if params.load_model_quant:
            self.dec_rnn = ops.RNN.RNN(
                params.rnn_d_model, 
                params.rnn_hidden_size,  
                nonlinearity='relu',
                bias=True, 
                batch_first=True,
                bidirectional=params.rnn_bidirectional,
                
                forward_w_ih_data=f"{weights_dir}/dec_rnn_forward_weights_weight_ih.npy",
                forward_w_ih_scale=f"{weights_dir}/dec_rnn_forward_weights_weight_ih_scale.npy",
                forward_w_hh_data=f"{weights_dir}/dec_rnn_forward_weights_weight_hh.npy",
                forward_w_hh_scale=f"{weights_dir}/dec_rnn_forward_weights_weight_hh_scale.npy",
                
                inverse_w_ih_data=f"{weights_dir}/dec_rnn_inverse_weights_weight_ih.npy",
                inverse_w_ih_scale=f"{weights_dir}/dec_rnn_inverse_weights_weight_ih_scale.npy",
                inverse_w_hh_data=f"{weights_dir}/dec_rnn_inverse_weights_weight_hh.npy",
                inverse_w_hh_scale=f"{weights_dir}/dec_rnn_inverse_weights_weight_hh_scale.npy",
                
                forward_b_ih=f"{weights_dir}/dec_rnn_forward_bias_bias_ih.npy",
                forward_b_hh=f"{weights_dir}/dec_rnn_forward_bias_bias_hh.npy",
                
                inverse_b_ih=f"{weights_dir}/dec_rnn_inverse_bias_bias_ih.npy",
                inverse_b_hh=f"{weights_dir}/dec_rnn_inverse_bias_bias_hh.npy")
        else:
            self.dec_rnn = ops.RNN.RNN(
                params.rnn_d_model, 
                params.rnn_hidden_size,  
                nonlinearity='relu',
                bias=True, 
                batch_first=True,
                bidirectional=params.rnn_bidirectional,
                
                forward_w_ih_data=f"{weights_dir}/dec_rnn_forward_weight_ih.npy",
                forward_w_hh_data=f"{weights_dir}/dec_rnn_forward_weight_hh.npy",
                inverse_w_ih_data=f"{weights_dir}/dec_rnn_inverse_weight_ih.npy",
                inverse_w_hh_data=f"{weights_dir}/dec_rnn_inverse_weight_hh.npy",
                
                forward_b_ih=f"{weights_dir}/dec_rnn_forward_bias_ih.npy",
                forward_b_hh=f"{weights_dir}/dec_rnn_forward_bias_hh.npy",
                inverse_b_ih=f"{weights_dir}/dec_rnn_inverse_bias_ih.npy",
                inverse_b_hh=f"{weights_dir}/dec_rnn_inverse_bias_hh.npy")
        
        rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
        if params.load_model_quant:
            self.dec_output = ops.Linear.Linear(rnn_hidden_size_factor*params.rnn_hidden_size, 
                                    params.nlp_output_size,
                                    weights_data=f"{weights_dir}/dec_output_weights.npy",
                                    weights_data_scale=f"{weights_dir}/dec_output_weights_scale.npy",
                                    bias_data=f"{weights_dir}/dec_output_bias.npy")
        else:
            self.dec_output = ops.Linear.Linear(rnn_hidden_size_factor*params.rnn_hidden_size, 
                                    params.nlp_output_size,
                                    weights_data=f"{weights_dir}/onnx__MatMul_83.npy",
                                    bias_data=f"{weights_dir}/dec_output_bias.npy")
        
    def forward(self, x, h): 
        
        x, h  = self.dec_rnn(x, h)

        x = self.dec_output(x)
        
        x = np.squeeze(x, 2)
        
        return x, h