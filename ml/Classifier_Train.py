import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from LR import LR
from KNN import KNN
from SVM import SVM
from XGBoost import XGBoost
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))
from lib.Params import Params
from lib.Classifier_Dataset import PthDataset
sys.path.pop()

def main():
    global params
    params = Params()
        
    # data loader
    train_dataset = PthDataset(file_path='../data/classifier_train_set.pth')
    test_dataset = PthDataset(file_path='../data/classifier_test_set.pth')
    val_dataset = PthDataset(file_path='../data/classifier_validate_set.pth')
    
    X_train, y_train = train_dataset.data.numpy().reshape(-1, 6), train_dataset.label.numpy().reshape(-1)
    X_test,  y_test  = test_dataset.data.numpy().reshape(-1, 6),  test_dataset.label.numpy().reshape(-1)
    X_val,   y_val   = val_dataset.data.numpy().reshape(-1, 6),   val_dataset.label.numpy().reshape(-1)

    # model
    model_file = None
    if params.model_arch == "lr":
        model = LR(params)
        model_file = "lr_model.joblib"
    elif params.model_arch == "knn":
        model = KNN(params)
        model_file = "knn_model.joblib"
    elif params.model_arch == "svm":
        model = SVM(params)
        model_file = "svm_model.joblib"
    elif params.model_arch == "xgboost":
        model = XGBoost(params)
        model_file = "xgb_model.json"
    
    # model dir
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
        
    model_path = f"{params.model_dir}/{model_file}"
    
    model.fit(X_train, y_train, X_test, y_test)
    
    y_val_pred = model.decode(len(X_val), X_val)
    ber = (np.sum(np.abs(y_val - y_val_pred))/y_val.shape[0])
    print('Validation ber: {}'.format(ber))
    
    model.save_model(model_path)
    
    if params.model_arch == "xgboost":
        model.feature_importance()
        
if __name__ == '__main__':
    main()