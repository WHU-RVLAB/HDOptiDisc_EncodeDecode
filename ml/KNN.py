import sys
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Utils import codeword_threshold
sys.path.pop()

class KNN(object):
    
    def __init__(self, params:Params):
        self.params = params
        
    def fit(self, X_train, y_train, X_test, y_test):
        knn_model = KNeighborsClassifier(n_neighbors=8, algorithm='ball_tree')
        knn_model.fit(X_train, y_train)
        self.knn_model = knn_model

    def decode(self, eval_length, X_val):
        y_pred = self.knn_model.predict(X_val)
        y_val = codeword_threshold(y_pred)[:eval_length]
        y_val = np.array(y_val).reshape(1, -1)
        return y_val
    
    def save_model(self, model_file):
        dump(self.knn_model, model_file)
        
    def load_model(self, model_file):
        self.knn_model = load(model_file)
    
