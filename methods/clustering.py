import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional

import numpy as np

from src.base import BaseEstimator
from src.metrics import evaluate_clustering_model


class KMeans(BaseEstimator):
    def __init__(self, n_clusters: int = 2, max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.rng_ = np.random.RandomState(random_state)
        self.centroids_ = None

    def fit(self, X):
        # initialize centroids
        self.centroids_ = X[self.rng_.choice(X.shape[0], self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            # assign closest centroids to each point
            dist = np.linalg.norm(X[:, None] - self.centroids_, axis=2)
            self.labels_ = np.argmin(dist, axis=1)

            # update centroids
            new_centroids = []
            for j in range(self.n_clusters):
                if X[self.labels_ == j].size > 0:
                    new_centroids.append(X[self.labels_ == j].mean(axis=0))
                else:
                    new_centroids.append(self.centroids_[j])
            new_centroids = np.array(new_centroids)
            
            # check convergence
            if np.all(np.abs(new_centroids - self.centroids_) < self.tol):
                break
            else:
                self.centroids_ = new_centroids
        else:
            print("KMeans did not converge in {} iterations.".format(self.max_iter))
            
        return self

    def predict(self, X):
        # predict cluster labels for input data
        dist = np.linalg.norm(X[:, None] - self.centroids_, axis=2)
        labels = np.argmin(dist, axis=1)
        return labels
    
    def score(self, X_test):
        # evaluate model performance on test data
        performance = evaluate_clustering_model(self, X_test)
        print("KMeans performance on test data:")
        print(performance)
