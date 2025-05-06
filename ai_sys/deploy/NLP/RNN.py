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
        weights_dir = './weights'
        self.dec_rnn = ops.RNN.RNN(
            params.rnn_d_model, 
            params.rnn_hidden_size,  
            nonlinearity='relu',
            bias=True, 
            batch_first=True,
            bidirectional=params.rnn_bidirectional,
            weight_ih_data=f"{weights_dir}/onnx__RNN_52.npy",
            weight_hh_data=f"{weights_dir}/onnx__RNN_53.npy",
            bias_data=f"{weights_dir}/onnx__RNN_51.npy")
        
        rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
        self.dec_output = ops.Linear.Linear(rnn_hidden_size_factor*params.rnn_hidden_size, 
                                 params.nlp_output_size,
                                 weights_data=f"{weights_dir}/onnx__MatMul_54.npy",
                                 bias_data=f"{weights_dir}/dec_output_bias.npy")
        
    def forward(self, x, h): 
        
        x, h  = self.dec_rnn(x, h)

        x = self.dec_output(x)
        
        x = np.squeeze(x, 2)
        
        return x, h