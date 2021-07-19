import os
import json
import time
import warnings
import requests
import numpy as np
from sklearn.exceptions import DataConversionWarning

from util import load_dnn_data
from variables import host, port, inference_data_path

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

URL = "http://{}:{}/predict".format(host, port)

def filter_data(n_samples=10):
    if not os.path.exists(inference_data_path):
        X, _ , Y, _ , _ = load_dnn_data()
        idxs = Y.argsort()
        X = X[idxs,:]
        Y = Y[idxs]    

        n_count = list(set(Y))
        
        for j in n_count:
            idxs = Y == j
            X_j = X[idxs,:]
            Y_j = Y[idxs]
            X_j = X_j[:n_samples,:]
            Y_j = Y_j[:n_samples]
            
            if j==0:
                X_all = X_j
                Y_all = Y_j
            else:
                X_all = np.concatenate((X_all, X_j), axis=0)
                Y_all = np.concatenate((Y_all, Y_j), axis=0)

        np.savez(inference_data_path, name1=X_all, name2=Y_all)

    else:
        data = np.load(inference_data_path)
        X_all = data["name1"]
        Y_all = data["name2"]
    return X_all, Y_all
        
def post_data():
    X, Y = filter_data()
    for (x, y) in zip(X, Y):
        time.sleep(0.04)
        x = x.tolist()
        y = int(y)
        data = {
                "csi data" : str(x),
                "true count" : str(y)
                }
        res = requests.post(URL, data=json.dumps(data))

post_data()

