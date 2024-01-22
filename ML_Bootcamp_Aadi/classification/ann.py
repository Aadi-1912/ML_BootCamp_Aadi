import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from classification.utils import *


class HiddenLayer:
    def __init__(self, n_neurons=5):
        self.n_neurons = n_neurons

    def feed(self, X):
        self.X = X
        self.weights = 0.1*np.random.randn(self.n_neurons, self.X.shape[0])
        self.bias = np.random.rand(self.n_neurons, 1)
        self.out = self.weights.dot(X) + self.bias 



def initiate(X, neurons_each_layer):
    n_layers = len(neurons_each_layer)
    wnb_params = {}
    output = X
    
    for i in range(1, n_layers):
        layer = HiddenLayer(n_neurons=neurons_each_layer[i])
        layer.feed(output)
        output = layer.out
        wnb_params['layer'+str(i)] = layer

    return wnb_params




def propagate_forward(X, wnb_params):
   
    layer_feed = {}
    L = len(wnb_params)                  
    
    layer_feed['a_feed0'] = X

    for l in range(1, L):
        layer_feed['z_feed'+str(l)] = wnb_params['layer'+str(l)].weights.dot(layer_feed['a_feed'+str(l-1)]) + wnb_params['layer'+str(l)].bias
        layer_feed['a_feed'+str(l)] = relu(layer_feed['z_feed'+str(l)])
            

    layer_feed['z_feed'+str(L)] = wnb_params['layer'+str(L)].weights.dot(layer_feed['a_feed'+str(L-1)]) + wnb_params['layer'+str(L)].bias
    layer_feed['a_feed'+str(L)] = softmax(layer_feed['z_feed'+str(L)])
    
    return layer_feed['a_feed'+str(L)], layer_feed




def propagate_backward(y, wnb_params, layer_feed):
    
    grads = {}
    L = len(wnb_params)
    m = y.shape[1]
    
    grads["dl/dz"+str(L)] = layer_feed['a_feed'+str(L)] - y
    grads["dl/dw"+str(L)] = 1/m * np.dot(grads["dl/dz"+str(L)],layer_feed['a_feed'+str(L-1)].T)
    grads["dl/db"+str(L)] = 1/m * np.sum(grads["dl/dz"+str(L)], axis = 1, keepdims = True)
    
    for l in reversed(range(1, L)):
        grads["dl/dz"+str(l)] = np.dot(wnb_params['layer'+str(l+1)].weights.T,grads["dl/dz"+str(l+1)])*derivative_relu(layer_feed['a_feed'+str(l)])
            
        grads["dl/dw"+str(l)] = 1/m*np.dot(grads["dl/dz"+str(l)],layer_feed['a_feed'+str(l-1)].T)
        grads["dl/db"+str(l)] = 1/m*np.sum(grads["dl/dz"+str(l)], axis = 1, keepdims = True)

    return grads




def update_params(wnb_params, grads, learning_rate):

    L = len(wnb_params) 
    
    for l in range(L):
        wnb_params["layer"+str(l+1)].weights -= learning_rate * grads["dl/dw"+str(l+1)]
        wnb_params["layer"+str(l+1)].bias -= learning_rate * grads["dl/db"+str(l+1)]
        
    return wnb_params




class NeuralNetwork:
    def __init__(self, neurons_each_layer : list[int]):
        self.neurons_each_layer = neurons_each_layer

    def fit(self, X, y, learning_rate=0.1, n_iterations=1000):
        self.wnb_params = initiate(X, neurons_each_layer)
        L = len(self.wnb_params)//2
        m = y.shape[1]
        
        for i in range(0, n_iterations):
            AL, layer_feed = propagate_forward(X, self.wnb_params)
            cost = -(1/m)*np.sum(y*np.log(AL))
            grads = propagate_backward(y, self.wnb_params, layer_feed)
            self.wnb_params = update_params(self.wnb_params, grads, learning_rate)
            if i%(n_iterations/10) == 0:
                print(f'{i}th iteration done. Cost : {cost}')
        
    def predict(self,X_test):
        self.y_pred, _ = propagate_forward(X_test, self.wnb_params)
        return self.y_pred

    def check_performance(self,X_train, X_test, y_train, y_test):
        performance = {
            'Training Data' : [calc_accuracy(X_train, y_train, self.wnb_params)],
            'Testing Data' : [calc_accuracy(X_test, y_test, self.wnb_params)]
        }
        perf_df = pd.DataFrame(performance, index=['Accuracy'])
        return perf_df




