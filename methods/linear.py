import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from base import BaseRegressor, BaseClassifier
from metrics import evaluate_classification_model, evaluate_regression_model


class LinearRegressor(BaseRegressor):
    def __init__(self):
        self.weights_ = None

    def fit(self, X, y):
        # fit linear regression model to input data
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # add intercept term
        self.weights_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        return self
    
    def predict(self, X):
        # predict target values for input data
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # add intercept term
        y_pred = X.dot(self.weights_)
        return y_pred

    def score(self, X_test, y_test):

        # evaluate model performance on test data
        #performance = evaluate_regression_model(self, X_test, y_test)
        #print("LinearRegressor performance on test data:")
        #print(performance)

        return super().score(X_test, y_test)


class SGDRegression(BaseRegressor):
    def __init__(self, eta: float = 0.001, n_iterations: int = 1000, batch_size: int = 1, bias: bool = False):
        super(SGDRegression).__init__()
        self.bias_ = None
        self.weights_ = None
        self.eta = eta
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.bias = bias
    
    def fit(self, X, y):
        # fit SGD regression model to input data
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))  # add intercept term
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)

        for i in range(self.n_iterations):
            batch_indices = np.random.choice(n_samples, size=self.batch_size, replace=False)
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            y_pred = X_batch.dot(self.weights_)
            error = y_pred - y_batch
            grad = X_batch.T.dot(error) / self.batch_size
            self.weights_ -= self.eta * grad

        return self

    def predict(self, X):
        # predict target values for input data
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))  # add intercept term
        y_pred = X.dot(self.weights_)
        return y_pred

    def score(self, X_test, y_test):

       # evaluate model performance on test data
        #performance = evaluate_regression_model(self, X_test, y_test)
        #print("SGDRegression performance on test data:")
        #print(performance)

        return super().score(X_test, y_test)


class LogisticRegression(BaseClassifier):
    """
    Logistic Regression.
    """

    def __init__(self, learning_rate: float = 0.001, num_epochs: int = 100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.array([])
        self.intercept = 0

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.intercept = 0

        for _ in range(self.num_epochs):
            linear_model = np.dot(X, self.weights) + self.intercept
            y_predicted = self._sigmoid(linear_model)

            delta_weights = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            delta_intercept = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * delta_weights
            self.intercept -= self.learning_rate * delta_intercept

        return self

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.intercept
        y_predicted = self._sigmoid(linear_model)
        return np.column_stack((1 - y_predicted, y_predicted))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.intercept
        y_predicted = self._sigmoid(linear_model)
        y_predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_classes)

    def score(self, X_test, y_test):

        # evaluate model performance on test data
        #performance = evaluate_classification_model(self, X_test, y_test)
        #print("LogisticRegression performance on test data:")
        #print(performance)

        return super().score(X_test, y_test)
