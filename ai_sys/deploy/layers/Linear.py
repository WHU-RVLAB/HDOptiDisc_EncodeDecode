import numpy as np

class Linear:
    def __init__(self, in_features, out_features, weights_data, bias_data):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        limit = 1 / np.sqrt(in_features)
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features))
        self.bias = np.random.uniform(-limit, limit, (out_features,)) if bias else None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x: [batch_size, in_features]
        # weight: [out_features, in_features]
        # bias: [out_features]
        output = x @ self.weight.T
        if self.use_bias:
            output += self.bias
        return output
