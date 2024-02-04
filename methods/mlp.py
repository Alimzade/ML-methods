"""
Module for the Multi-Layer Perceptron (MLP) regressor (Session 10).

"""
from collections.abc import Sequence
from typing import Optional
import numpy as np

from src.base import BaseRegressor


class ModularLinearLayer:
    """
    Modular linear layer of a neural network with L2 regularization term and bias.
    This layer has its own backward and update function, which are needed for use in a modular MLP.

    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        weight_: Weight matrix of shape (in_features, out_features)
        bias_: Bias vector of shape (out_features,)
        _weight_grad: Gradient of the weight matrix
        _bias_grad: Gradient of the bias vector
        _prev_input: Input to the layer, used for backpropagation
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.RandomState = np.random.RandomState()):
        self._bias_grad = None
        self._weight_grad = None
        self.in_features = in_features
        self.out_features = out_features
        self.weight_ = rng.randn(in_features, out_features)
        self.bias_ = rng.randn(out_features, 1).squeeze(axis=1)
        self._prev_input = None

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
        self._prev_input = X
        output = np.dot(X, self.weight_) + self.bias_
        return output

    def backward(self, upstream_grad: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Backward pass of the layer - only needed for Modular MLPs.
        Parameters
        ----------
        upstream_grad : np.ndarray
            Gradient of the loss w.r.t. the output of the layer
        alpha : Optional[float]
            Regularization parameter

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. the input of the layer

        """
        # TODO Implement backward pass
        # derivative of Cost w.r.t weight
        self._weight_grad = np.dot(self._prev_input.T, upstream_grad)
        
        # Add regularization terms for weights
        if alpha is not None:
            self._weight_grad += alpha * self.weight_

        # derivative of Cost w.r.t bias, sum across rows
        self._bias_grad = np.sum(upstream_grad, axis=0)

        # derivative of Cost w.r.t _prev_input
        grad_input = np.dot(upstream_grad, self.weight_.T)
        return grad_input

    def update(self, lr: float) -> None:
        """
        Update the parameters of the layer - only needed for Modular MLPs.

        Parameters
        ----------
        lr : float
            Learning rate for the update
        """
        # TODO Implement weight update
        self.weight_ -= lr * self._weight_grad
        self.bias_ -= lr * self._bias_grad


class TanhLayer:
    """
    Layer for tanh activation function.
    """

    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        self._prev_result = np.tanh(X)
        return self._prev_result

    def backward(self, upstream_grad):
        """
        Compute the gradient of the ReLU activation function.

        Parameters
        ----------
        upstream_grad : np.ndarray
            The gradient of the cost function w.r.t. the output of the ReLU layer.

        Returns
        -------
        np.ndarray
            The gradient of the cost function w.r.t. the input of the ReLU layer.

        """
        # TODO Implement backward pass
        grad_input = (1 - np.square(self._prev_result)) * upstream_grad
        return grad_input


class MLPRegressor(BaseRegressor):
    """
    Multi-layer perceptron regressor with modular layer definition, configurable activation and L2 regularization.

    Parameters
    ----------
    hidden_layer_sizes : Sequence[int]
        Number of neurons in each hidden layer
    lr : float
        Learning rate
    epochs : int
        Number of epochs to train for
    random_state : Optional[int]
        Random seed for layer initialization
    alpha : float
        Regularization parameter
    activation : Optional[str]
        Activation function to use. Can be 'relu', 'tanh' or 'sigmoid'
    """

    def __init__(self, hidden_layer_sizes: Sequence[int] = (3, 5, 3), lr: float = 0.01, epochs: int = 100,
                 random_state: Optional[int] = None, alpha: float = 0.0001, activation: Optional[str] = 'tanh'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.layers_ = None
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.alpha = alpha
        self.activation = activation

    def fit(self, X: np.array, y: np.array):
        in_size = X.shape[1]

        self.layers_ = []
        for size in self.hidden_layer_sizes:
            self.layers_.append(ModularLinearLayer(in_size, size, self.rng))
            in_size = size

            if self.activation == 'tanh':
                self.layers_.append(TanhLayer())

        self.layers_.append(ModularLinearLayer(in_size, 1, self.rng))

        # Gradient descent. For each batch...
        for epoch in range(0, self.epochs):
            # TODO Implement gradient descent
            epoch_loss = 0.0

            # For each sample
            for i in range(X.shape[0]):
                # Forward pass
                output = X[i].reshape(1, -1)
                for layer in self.layers_:
                    output = layer(output)
            
                # Calculate loss and its gradient
                loss = (output - y[i])**2
                epoch_loss += loss.item()  # Accumulate the loss value
                grad = 2 * (output - y[i])

                # Backpropagation of gradients
                for layer in reversed(self.layers_):
                    grad = layer.backward(grad, self.alpha)

                # Update weights and biases in each layer according to their gradients
                for layer in self.layers_:
                    if isinstance(layer, ModularLinearLayer):
                        layer.update(self.lr)
                        
                #print(f"Sample {i+1} - Output: {output}, Loss: {loss}, Gradient: {grad}")

            # Calculate average loss per sample
            avg_loss = epoch_loss / X.shape[0]
            print(f"Epoch {epoch+1}: Loss = {avg_loss}")

        return self
    
    def predict(self, X: np.array) -> np.array:
        predictions = []  # List to store the predicted outputs for each input sample

        for i in range(X.shape[0]):
            # Forward pass through the layers
            output = X[i].reshape(1, -1)  # Reshape the input sample (to 2D)

            # Apply each layer's forward pass operation
            for layer in self.layers_:
                output = layer(output)

            # Store the predicted output for the current sample
            predictions.append(output)

        # Convert the list to a numpy array and squeeze the dimensions (single output value)
        return np.array(predictions).squeeze()