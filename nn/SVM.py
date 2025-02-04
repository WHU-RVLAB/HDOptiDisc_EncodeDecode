import sys
import os
import numpy as np
from sklearn.svm import SVC
from joblib import dump, load

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import codeword_threshold
sys.path.pop()

class SVM(object):
    
    def __init__(self, params:Params):
        self.params = params
        
    def fit(self, X_train, y_train, X_test, y_test):
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_model.fit(X_train, y_train)
        self.svm_model = svm_model

    def decode(self, eval_length, X_val):
        y_pred = self.svm_model.predict(X_val)
        y_val = codeword_threshold(y_pred)[:eval_length]
        y_val = np.array(y_val).reshape(1, -1)
        return y_val
    
    def save_model(self, model_file):
        dump(self.svm_model, model_file)
        
    def load_model(self, model_file):
        self.svm_model = load(model_file)
    
