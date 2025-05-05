import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)))
from BaseModel import BaseModel
sys.path.pop()

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from layers.Linear import Linear
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
        weights_dir = '../weights'
        self.dec_input = Linear(params.nlp_input_size, 
                                params.rnn_d_model,
                                weights_data=f"{weights_dir}/dec_input_weights.bin",
                                bias_data=f"{weights_dir}/dec_input_bias.bin")
        self.dec_rnn = nn.GRU(params.rnn_d_model, 
                                    params.rnn_hidden_size, 
                                    params.rnn_layer, 
                                    bias=True, 
                                    batch_first=True,
                                    dropout=params.rnn_dropout_ratio, 
                                    bidirectional=params.rnn_bidirectional)
        
        rnn_hidden_size_factor = 2 if params.rnn_bidirectional else 1
        self.dec_output = Linear(rnn_hidden_size_factor*params.rnn_hidden_size, 
                                 params.nlp_output_size,
                                 weights_data=f"{weights_dir}/dec_output_weights.bin",
                                 bias_data=f"{weights_dir}/dec_output_bias.bin")
        
    def forward(self, x, h): 
        
        x = self.dec_input(x)
        
        x, h  = self.dec_rnn(x, h)

        x = self.dec_output(x)
        
        x = torch.sigmoid(x)
        
        x = torch.squeeze(x, 2)
        
        return x, h