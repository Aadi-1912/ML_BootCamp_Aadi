import numpy as np
import pandas as pd



class Standarize:
    
    mean = std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0) + 1e-8
        self.std = X.std(axis=0) + 1e-8
        
        
    def fit_transform(self, X):
        self.fit(X)
        X_scaled = (X - self.mean)/self.std
        return X_scaled
        

    def transform(self, X):
        try:
            X_scaled = (X - self.mean)/self.std
            return X_scaled
        except TypeError:
            raise TypeError('No data has been provided to calculate mean and standard deviation')






def step(z):
    return 1 if z >= 0 else 0




def sigmoid(z):
    return 1/(1 + np.exp(-z))




def softmax(z):
    expZ = np.exp(z)
    return expZ/(np.sum(expZ, 0))





def one_hot_encoding(y):
    classes = np.unique(y)
    y_encoded = np.zeros(y.shape[0])
    
    for i in classes:
        y_df = pd.DataFrame(y).loc[y == i].copy()
        y_n = pd.DataFrame(np.zeros(y.shape[0]))
        y_n.loc[y_df.index] = 1
        y_encoded = np.c_[y_encoded, y_n]
    
    y_encoded = np.delete(y_encoded, 0, 1)
    
    return y_encoded




def relu(Z):
    A = np.maximum(0,Z)
    return A




def derivative_relu(Z):
    return np.array(Z > 0, dtype = 'float')