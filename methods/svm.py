import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.base import BaseClassifier
import numpy as np


class LinearSVM(BaseClassifier):
    def __init__(self, lr: float = 1.0, tol: float = 1e-4, max_iter: int = 1000):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        self.weights = None
        self.bias = None
        
        num_samples, num_features = X.shape

        # Initialize the weight vector and bias term
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Perform gradient descent to optimize the SVM model
        for _ in range(self.max_iter):
            # Compute the decision scores
            scores = np.dot(X, self.weights) + self.bias

            # Compute the hinge loss for misclassified samples
            margins = y * scores
            hinge_loss = np.maximum(0, 1 - margins)

            # Compute the gradient of the loss function
            gradient_weights = np.zeros(num_features)
            gradient_bias = 0
            mask = margins < 1
            gradient_weights -= np.dot(mask * y, X)
            gradient_bias -= np.sum(mask * y)

            # Update the weight vector and bias term using gradient descent
            self.weights -= self.lr * gradient_weights
            self.bias -= self.lr * gradient_bias

            # Check for convergence based on tolerance
            if np.linalg.norm(self.lr * gradient_weights) < self.tol:
                break

    def predict(self, X):
        # Compute the decision scores
        scores = np.dot(X, self.weights) + self.bias

        # Assign class labels based on the sign of the scores
        predictions = np.sign(scores)

        return predictions
