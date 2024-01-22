import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from classification.utils import Standarize 



class KNearestNeighbor:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.y_preds = np.empty(0)
        for i in range(len(X_test)):
            x = X_test[i]
            distance = np.linalg.norm(self.X_train - x, axis=1)
            indices = np.argsort(distance)[:self.k]
            labels = self.y_train[indices]
            y_pred = max(labels, key=lambda i: np.count_nonzero(labels == i))
            self.y_preds = np.append(self.y_preds, y_pred)
            if i%(len(X_test)//10) == 0:
                print(i)

        return self.y_preds

    def calc_accuracy(self, y_test):
        y_total = len(y_test) 
        y_correct = np.count_nonzero(y_test == self.y_preds)
        accuracy = y_correct/y_total
        return accuracy




def calc_accuracy(y_test, y_preds):
    y_total = len(y_test) 
    y_correct = np.count_nonzero(y_test == y_preds)
    accuracy = y_correct/y_total
    return accuracy
    
