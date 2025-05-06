import numpy as np

class RNN:
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 nonlinearity='relu', 
                 bias=True, 
                 batch_first=True,
                 bidirectional=True,
                 weight_ih_data=None,
                 weight_hh_data=None,
                 bias_data=None):
        assert nonlinearity == 'relu', "Only relu is supported"
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Load weights and reshape
        self.weight_ih = np.load(weight_ih_data).reshape(self.num_directions, hidden_size, input_size)
        self.weight_hh = np.load(weight_hh_data).reshape(self.num_directions, hidden_size, hidden_size)
        
        if bias:
            bias = np.load(bias_data).reshape(self.num_directions, hidden_size * 2)
            self.bias_ih = bias[:, :hidden_size]
            self.bias_hh = bias[:, hidden_size:]

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
                h_t = self.relu(
                    step_input @ self.weight_ih[direction].T + self.bias_ih[direction] +
                    h_t @ self.weight_hh[direction].T + self.bias_hh[direction]
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
