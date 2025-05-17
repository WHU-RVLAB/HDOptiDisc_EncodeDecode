import numpy as np

class Linear:
    def __init__(self, in_features, out_features, weights_data, bias_data, weights_data_scale=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.load(weights_data).reshape(out_features, in_features)
        self.weight_scale = np.load(weights_data_scale) if weights_data_scale else 1.0
        self.bias = np.load(bias_data).reshape(out_features)

    def __call__(self, x):
        return (x @ self.weight.T)*self.weight_scale + self.bias
