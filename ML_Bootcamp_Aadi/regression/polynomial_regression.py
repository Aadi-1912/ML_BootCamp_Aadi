import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path
sys.path.append(parent_dir)

from regression.utils import train_test_split, clean_data, Standarize, r2_score, poly_transformation
import numpy as np
import pandas as pd

def grad_desc(X_poly, y_train, learning_rate=0.01, n_iterations=1000):
    order = X_poly.shape
    m, n = order[0], order[1]
    
    theta = np.zeros((n, 1))

    for i in range(n_iterations):
        # It will calculate the predicted values of the model.
        y_pred = X_poly.dot(theta)

        # It will calculate the the residuals in predictions.
        residuals = y_pred - y_train

        # multiplying the coefficients of theta as per the chain rule of differentiation and then multiplying with 1/2m. This will give the gradient of the cost function.
        gradients = (1/m)*X_poly.T.dot(residuals)

        step_size = gradients*learning_rate

        theta -= step_size

        if i%1000 == 0: print(f'{i}th iteration done. Cost = {1/(2*m)*(residuals**2).sum().iloc[0]}')

    return theta, residuals
        






class Linear_regression:
    '''
    A class to perform Linear(and Polynomial) Regression depending on the value of the degree provided.
    Data should be first standarize before training and testing the regression model.

    ...
    Attributes
    ----------

    degree: int
        Degree that to be specified for Polynomial Linear Regression, by default degree = 1.

    theta: numpy.ndarray[float]
        Array containing Bias intercept and Coefficient terms.

    residuals: numpy.ndarray[float]
        Array containing difference between the predicted values and the actual target values of y.

    min_cost: float
        Minimum value cost function of the model.

    y_train_pred: numpy.ndarray[float]
        Array containing predicted values of trained model(y_train).

    y_test_pred: numpy.ndarray[float]
        Array containing predicted values of testes model(y_test).
        

    Methods
    -------
    fit(X_train, y_train, learning_rate = 0.01, n_iterations = 1000)
        Will train the model.
    
    predict(X_test)
        will return the probable values of y.
    
    '''

    def __init__(self, degree=1):
        self.degree = degree
        
    
    def fit(self, X_train, y_train, learning_rate = 0.01, n_iterations = 1000):
        '''
        Equation of the line is given by:-
        
        y_pred = theta_0 + (theta_1)x + (theta_2)x^2 + ...

        The method will calculate the values of theta_0, theta_1 and theta_2 using gradient descent method.
        '''

        # X_poly = poly_transformation(X_train, degree=self.degree)

        self.theta, self.residuals = grad_desc(X_poly, y_train, learning_rate=learning_rate, n_iterations=n_iterations)

        # Cost_Function(J) = 1/(2m) * summation((y_pred-y_actual)^2)
        self.min_cost = 1/2*((self.residuals)**2).mean()

        # Predicted values of already trained data
        self.y_train_pred = X_poly.dot(self.theta)
        

    def predict(self, X_test):
        '''This method will predict the values of y for the given values of X using the coefficients of the trained model.'''
        # X_poly = poly_transformation(X_test, degree=self.degree)
        self.y_test_pred = X_poly.dot(self.theta)

        return self.y_test_pred
        

    def check_performance(self, y_train, y_test):
        '''Displays the performance of trained and tested model'''
        performance = {
            'Training Data':[r2_score(y_train, self.y_train_pred)],
            'Testing Data':[r2_score(y_test, self.y_test_pred)]
        }
        perf_df = pd.DataFrame(performance, index=['R2 Score'])
        return perf_df


