import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from classification.utils import Standarize, step, sigmoid, one_hot_encoding


def grad_desc(X, y, learning_rate=0.01, n_iterations=100):
    m, n = X.shape
    w = np.zeros(n)
    
    for i in range(1, n_iterations+1):
        
        z = X.dot(w)
        y_pred = sigmoid(z)
        w += learning_rate*(y - y_pred).T.dot(X)/m
        
        if i%10 == 0:
            loss_func = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))/m
            print(f'{i}th iteration done. Loss = {loss_func}')

    return w






def calc_accuracy(y_test, y_preds):
    m, n = y_test.shape
    y_total = m*n
    
    y_bool = (y_test == y_preds)
    y_correct = np.count_nonzero(y_bool == True)
    
    accuracy = y_correct/y_total
    return accuracy





class LogisticRegression:
    def fit(self, X_train, y_train, learning_rate=0.01, n_iterations=100):
        self.w_s = np.zeros((X_train.shape[1], 1))
        for k in range(y_train.shape[1]):
            y_train_col = y_train[:, k]
            w = grad_desc(X_train, y_train_col, learning_rate=learning_rate, n_iterations=n_iterations)
            self.w_s = np.insert(self.w_s, -1, w, axis=1)
            print(w.shape)
        self.w_s = np.delete(self.w_s, -1, axis=1)

    def predict(self, X_test):
        step_brute = np.vectorize(step)
        self.y_pred = step_brute(X_test.dot(self.w_s))
        return self.y_pred
        
