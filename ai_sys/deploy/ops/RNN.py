import numpy as np

class RNN:
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 
                forward_w_ih_data,
                forward_w_hh_data,
                inverse_w_ih_data,
                inverse_w_hh_data,
                
                
                forward_b_ih,
                forward_b_hh,
                inverse_b_ih,
                inverse_b_hh,
                
                forward_w_ih_scale=None,
                forward_w_hh_scale=None,
                inverse_w_ih_scale=None,
                inverse_w_hh_scale=None,
                
                 nonlinearity='relu', 
                 bias=True, 
                 batch_first=True,
                 bidirectional=True):
        assert nonlinearity == 'relu', "Only relu is supported"
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Load weights and reshape
        self.forward_w_ih = np.load(forward_w_ih_data).reshape(hidden_size, input_size)
        self.forward_w_hh = np.load(forward_w_hh_data).reshape(hidden_size, hidden_size)
        self.inverse_w_ih = np.load(inverse_w_ih_data).reshape(hidden_size, input_size)
        self.inverse_w_hh = np.load(inverse_w_hh_data).reshape(hidden_size, hidden_size)
        
        self.forward_w_ih_scale = np.load(forward_w_ih_scale) if forward_w_ih_scale else 1.0
        self.forward_w_hh_scale = np.load(forward_w_hh_scale) if forward_w_hh_scale else 1.0
        self.inverse_w_ih_scale = np.load(inverse_w_ih_scale) if inverse_w_ih_scale else 1.0
        self.inverse_w_hh_scale = np.load(inverse_w_hh_scale) if inverse_w_hh_scale else 1.0
        
        if bias:
            self.forward_b_ih = np.load(forward_b_ih).reshape(input_size, hidden_size)
            self.forward_b_hh = np.load(forward_b_hh).reshape(input_size, hidden_size)
            self.inverse_b_ih = np.load(inverse_b_ih).reshape(input_size, hidden_size)
            self.inverse_b_hh = np.load(inverse_b_hh).reshape(input_size, hidden_size)

    def relu(self, x):
        return np.maximum(0, x)

    def __call__(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(1, 0, 2)  # [T, B, input_size]
        
        seq_len, batch_size, _ = x.shape

        if h_0 is None:
            h_0 = np.zeros((self.num_directions, batch_size, self.hidden_size))

        outputs = []
        h_n = np.zeros_like(h_0)

        for direction in range(self.num_directions):
            h_t = h_0[direction]
            input_seq = x if direction == 0 else x[::-1]
            step_outputs = []

            for t in range(seq_len):
                step_input = input_seq[t]
                
                if direction == 0:
                    weight_ih = self.forward_w_ih
                    weight_ih_scale = self.forward_w_ih_scale
                    
                    bias_ih = self.forward_b_ih
                    
                    weight_hh = self.forward_w_hh
                    weight_hh_scale = self.forward_w_hh_scale
                    
                    bias_hh = self.forward_b_hh
                elif direction == 1:
                    weight_ih = self.inverse_w_ih
                    weight_ih_scale = self.inverse_w_ih_scale
                    
                    bias_ih = self.inverse_b_ih
                    
                    weight_hh = self.inverse_w_hh
                    weight_hh_scale = self.inverse_w_hh_scale
                    
                    bias_hh = self.inverse_b_hh
                h_t = self.relu(
                    (step_input @ weight_ih.T)*weight_ih_scale + bias_ih +
                    (h_t @ weight_hh.T)*weight_hh_scale + bias_hh
                )
                step_outputs.append(h_t)

            if direction == 1:
                step_outputs = step_outputs[::-1]
            
            outputs.append(np.stack(step_outputs, axis=0))  # [T, B, H]
            h_n[direction] = h_t

        if self.bidirectional:
            output = np.concatenate(outputs, axis=2)  # [T, B, 2H]
        else:
            output = outputs[0]  # [T, B, H]

        if self.batch_first:
            output = output.transpose(1, 0, 2)  # [B, T, H or 2H]

        return output, h_n
