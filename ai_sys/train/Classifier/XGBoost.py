import sys
import os
import numpy as np
from matplotlib import pyplot
import xgboost as xgb
from xgboost import plot_importance

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))))
from lib.Params import Params
from lib.Utils import codeword_threshold
sys.path.pop()

class XGBoost(object):
    
    def __init__(self, params:Params):
        self.params = params
        
    def fit(self, X_train, y_train, X_test, y_test):
        params = {
            'objective':'binary:logistic',
            'eta': 0.1,                    
            'n_estimators': 1000,          
            'max_depth': 5,
            'subsample': 0.8,              
            'colsample_bytree': 0.8,      
            'gamma': 0.1,                  
            'reg_alpha': 0.5,             
            'reg_lambda': 1.0,
            'early_stopping_rounds':50
        }
        eval_set = [(X_test, y_test)]
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(
                    X_train, 
                    y_train, 
                    eval_set=eval_set,
                    verbose=True)
        self.xgb_model = xgb_model

    def decode(self, X_val):
        if isinstance(self.xgb_model, xgb.Booster):
            y_pred = self.xgb_model.predict(xgb.DMatrix(X_val))
        else:
            y_pred = self.xgb_model.predict(X_val)
        y_val = codeword_threshold(y_pred)
        y_val = np.array(y_val).reshape(1, -1)
        return y_val
    
    def save_model(self, model_file):
        self.xgb_model.save_model(model_file)
        
    def load_model(self, model_file):
        self.xgb_model = xgb.Booster(model_file=model_file)
    
    def feature_importance(self):
        plot_importance(self.xgb_model)
        pyplot.show()
    
