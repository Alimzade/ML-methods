"""
Module for the Single-layer Perceptron Classifier with one hidden layer and tanh activation function (Session 09).
"""
from typing import Optional

import numpy as np

from src.base import BaseClassifier


class LinearLayer:
    """
    Linear layer of a neural network with bias.

    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        weight_: Weight matrix of shape (in_features, out_features)
        bias_: Bias vector of shape (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.RandomState = np.random.RandomState()):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_ = rng.randn(in_features, out_features)
        self.bias_ = rng.randn(out_features, 1).squeeze(axis=1)

    def __call__(self, X: np.array) -> np.array:
        """
        Forward pass of the layer
        Parameters
        ----------
        X : np.array
            Input to the layer of shape (batch_size, in_features)

        Returns
        -------
        np.array
            Output of the layer of shape (batch_size, out_features)

        """
        # TODO Implement forward pass
        return np.dot(X, self.weight_) + self.bias_


class SLPClassifier(BaseClassifier):
    """
    Single-layer Perceptron Classifier with one hidden layer and tanh activation function.

    Attributes:
        hidden_layer_size: Number of neurons in the hidden layer
        lr: Learning rate
        epochs: Number of epochs
        layer_: Linear layer of the hidden layer
        out_layer_: Linear layer of the output layer
        rng: Random number generator
        alpha: L2 regularization term
    """

    def __init__(self, hidden_layer_size: int = 3, lr: float = 0.01, epochs: int = 1000,
                 random_state: Optional[int] = None, alpha: float = 0.0001):
        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.epochs = epochs
        self.layer_: Optional[LinearLayer] = None
        self.out_layer_: Optional[LinearLayer] = None
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.alpha = alpha

    def fit(self, X: np.array, y: np.array):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        in_size = X.shape[1]
        out_size = y.shape[1]
        self.layer_ = LinearLayer(in_size, self.hidden_layer_size, self.rng)
        self.out_layer_ = LinearLayer(self.hidden_layer_size, out_size, self.rng)

        # one-hot encoding for labels
        y = np.eye(out_size, dtype=int)

        # Gradient descent. For each batch...
        for epoch in range(0, self.epochs):
            # TODO Implement gradient descent
            # Forward pass: calculate logits
            hidden = np.tanh(self.layer_(X))
            logits = self.out_layer_(hidden)

            # get softmax probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

            # Calculate loss and its gradient
            loss = -np.sum(y * np.log(probs))
            d_logits = probs - y
            
            # Backpropagation of gradients
            d_hidden = np.dot(d_logits, self.out_layer_.weight_.T) * (1 - hidden**2)
            d_layer_weight = np.dot(X.T, d_hidden)
            d_layer_bias = np.sum(d_hidden, axis=0)
            d_out_layer_weight = np.dot(hidden.T, d_logits)
            d_out_layer_bias = np.sum(d_logits, axis=0)

            # Update weights and biases in both layers according to their gradients
            self.layer_.weight_ -= self.lr * (d_layer_weight + self.alpha * self.layer_.weight_)
            self.layer_.bias_ -= self.lr * d_layer_bias
            self.out_layer_.weight_ -= self.lr * (d_out_layer_weight + self.alpha * self.out_layer_.weight_)
            self.out_layer_.bias_ -= self.lr * d_out_layer_bias
    
    def predict(self, X: np.array) -> np.array:
        """
        Make predictions for input samples.
        Parameters
        ----------
        X : np.array
            Input samples of shape (n_samples, n_features).

        Returns
        -------
        np.array
            Predicted class labels for the input samples.
        """
        # Perform forward pass to get the logits
        hidden = np.tanh(self.layer_(X))
        logits = self.out_layer_(hidden)

        # Apply softmax activation function to get class probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        # Convert probabilities to class labels
        predicted_labels = np.argmax(probs, axis=1)

        return predicted_labels
