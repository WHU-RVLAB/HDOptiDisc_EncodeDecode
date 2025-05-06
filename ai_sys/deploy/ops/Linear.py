import numpy as np

class Linear:
    def __init__(self, in_features, out_features, weights_data, bias_data):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.load(weights_data).reshape(out_features, in_features)
        self.bias = np.load(bias_data).reshape(out_features)

    def __call__(self, x):
        return x @ self.weight.T + self.bias
