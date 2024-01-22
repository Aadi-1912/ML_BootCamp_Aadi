import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class Standarize:
    
    mean = std = None
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        
        
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




def label(X, centroids, clusters):
    distances = np.zeros((X.shape[0], 1))
    for centroid in centroids:
        distance = np.sqrt(np.sum((X-centroid)**2, axis=1).reshape(X.shape[0], 1))
        distances = np.append(distances, distance, axis=1)

    distances = np.delete(distances, 0, axis=1)
    labels = np.argmin(distances, axis=1)
    return labels





def recentre(X, labels):
    centroids = np.zeros((1, X.shape[1]))
    for label in np.unique(labels):
        centroid = np.mean(X[labels == label], axis=0).reshape(1, X.shape[1])
        centroids = np.append(centroids, centroid, axis=0)

    centroids = np.delete(centroids, 0, axis=0)
    return centroids



class KMeans:
    inertia = 0
    
    def __init__(self, clusters):
        self.clusters = clusters

    def fit(self, X, n_iterations=100):
        self.centroids_cords = np.random.randint(0, len(X), self.clusters)
        self.centroids = X[self.centroids_cords]
        for _ in range(n_iterations):
            self.old_centroids = self.centroids
            self.labels = label(X, self.centroids, self.clusters)
            self.centroids = recentre(X, self.labels)
            if self.old_centroids.shape != self.centroids.shape:
                continue
            if (self.old_centroids == self.centroids).all():
                break
        
        for i in np.unique(self.labels):
            self.inertia += np.sum((X[self.labels == i] - self.centroids[i, :])**2)
            
        return self.labels
