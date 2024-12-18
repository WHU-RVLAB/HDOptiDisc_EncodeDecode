import numpy as np

class NRZI_Converter(object):
    
    def __init__(self):
        pass
    
    def forward_coding(self, z):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: x = (1 / 1 + D) z (mod 2)
        x_{-1} = 0
        '''
        
        length = np.size(z, 1)
        x = np.zeros((1, length))
        x[0, 0] = z[0, 0]
        for i in range(1, length):
            x[0, i] = x[0, i-1] + z[0, i]
        return x % 2
    
    def inverse_coding(self, x):
        '''
        Input: (1, length) array
        Output: (1, length) array
        Mapping: x = (1 + D) z (mod 2)
        z_{-1} = 0
        '''
    
        length = x.shape[1]
        z = np.zeros((1, length))
        z[0, 0] = x[0, 0]
        for i in range(1, length):
            z[0, i] = x[0, i] + x[0, i-1]
        return z % 2