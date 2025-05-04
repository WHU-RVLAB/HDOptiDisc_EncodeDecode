import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))
from lib.Params import Params
from lib.Utils import codeword_threshold
sys.path.pop()

class LR(object):
    
    def __init__(self, params:Params):
        self.params = params
        
    def fit(self, X_train, y_train, X_test, y_test):
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        self.lr_model = lr_model

    def decode(self, X_val):
        y_pred = self.lr_model.predict(X_val)
        y_val = codeword_threshold(y_pred)
        y_val = np.array(y_val).reshape(1, -1)
        return y_val
    
    def save_model(self, model_file):
        dump(self.lr_model, model_file)
        
    def load_model(self, model_file):
        self.lr_model = load(model_file)
    
