import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Module for the Convolutional Neural Network (CNN) classifier (Session 11).
"""

from typing import Optional, Tuple, Sequence, Union

import numpy as np

from numpy.lib.stride_tricks import as_strided
from numpy import einsum

from src.base import BaseClassifier
from src.mlp import ModularLinearLayer


class ConvLayer:
    """
    Convolutional layer of a convolutional neural network.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the kernel
        stride: Stride of the convolution
        padding: Padding of the convolution
        bias_: Bias vector of shape (out_channels,)
        weight_: Weight matrix of shape (out_channels, in_channels, kernel_size, kernel_size)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1), padding: int = 0,
                 rng: np.random.RandomState = np.random.RandomState()):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_ = rng.randn(out_channels, 1).squeeze()
        self.weight_ = rng.randn(*kernel_size, out_channels, in_channels)
        self._bias_grad = None
        self._weight_grad = None
        self._prev_input = None

    def __call__(self, X: np.array) -> np.array:
        """
        Forward pass of the layer
        Parameters
        ----------
        X : np.array
            Input to the layer of shape (batch_size, width, height, in_channels)

        Returns
        -------
        np.array
            Output of the layer of shape (batch_size, width_, height_, out_channels)

        """
        batch_size, in_height, in_width, _ = X.shape
        kernel_height, kernel_width = self.kernel_size
        out_height = (in_height - kernel_height + 2 * self.padding) // self.stride[0] + 1
        out_width = (in_width - kernel_width + 2 * self.padding) // self.stride[1] + 1

        # Add padding to input
        X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        # Perform convolution using stride trick
        strides = (X_padded.strides[0], self.stride[0] * X_padded.strides[1], self.stride[1] * X_padded.strides[2], X_padded.strides[3])
        patches = as_strided(X_padded, shape=(batch_size, out_height, out_width, kernel_height, kernel_width, self.in_channels), strides=strides)
        convolved = einsum('ijklmno,pklmno->ijp', patches, self.weight_)

        # Add bias
        convolved += self.bias_

        # Store previous input for backward pass
        self._prev_input = X

        return convolved

    def backward(self, upstream_grad: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Backward pass of the layer
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
        batch_size, in_height, in_width, _ = self._prev_input.shape
        kernel_height, kernel_width = self.kernel_size
        out_height = (in_height - kernel_height + 2 * self.padding) // self.stride[0] + 1
        out_width = (in_width - kernel_width + 2 * self.padding) // self.stride[1] + 1

        # Compute gradients with respect to weights and bias
        self._weight_grad = einsum('ijklmno,ijp->pklmno', as_strided(self._prev_input, shape=(batch_size, out_height, out_width, kernel_height, kernel_width, self.in_channels), strides=self._prev_input.strides), upstream_grad)
        self._bias_grad = np.sum(upstream_grad, axis=(0, 1))

        # Compute gradients with respect to input
        output_grad_padded = np.pad(upstream_grad, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        input_grad = np.zeros_like(self._prev_input)
        for i in range(out_height):
            for j in range(out_width):
                input_grad[:, i * self.stride[0]:i * self.stride[0] + kernel_height, j * self.stride[1]:j * self.stride[1] + kernel_width, :] += einsum('pklmno,ijp->ijklmno', self.weight_, output_grad_padded[:, i:i + 1, j:j + 1, :])

        # Add regularization terms for weights
        if alpha is not None:
            self._weight_grad += alpha * self.weight_

        return input_grad

    def update(self, lr: float) -> None:
        """
        Update the parameters of the layer

        Parameters
        ----------
        lr : float
            Learning rate for the update
        """
        self.weight_ -= lr * self._weight_grad
        self.bias_ -= lr * self._bias_grad


class PoolingLayer():
    """
    Pooling layer of a convolutional neural network.

    Attributes:
        kernel_size: Size of the kernel
        stride: Stride of the convolution
        padding: Padding of the convolution
    """

    def __init__(self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1), padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, X: np.array) -> np.array:
        """
        Forward pass of the layer
        Parameters
        ----------
        X : np.array
            Input to the layer of shape (batch_size, width, height, in_channels)

        Returns
        -------
        np.array
            Output of the layer of shape (batch_size, width, height, in_channels)

        """
        batch_size, in_height, in_width, in_channels = X.shape
        kernel_height, kernel_width = self.kernel_size
        out_height = (in_height - kernel_height + 2 * self.padding) // self.stride[0] + 1
        out_width = (in_width - kernel_width + 2 * self.padding) // self.stride[1] + 1

        # Add padding to input
        X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        # Perform pooling
        strides = (X_padded.strides[0], self.stride[0] * X_padded.strides[1], self.stride[1] * X_padded.strides[2], X_padded.strides[3])
        patches = as_strided(X_padded, shape=(batch_size, out_height, out_width, kernel_height, kernel_width, in_channels), strides=strides)
        pooled = np.max(patches, axis=(3, 4))

        return pooled

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer - only needed for Modular MLPs.
        Parameters
        ----------
        upstream_grad : np.ndarray
            Gradient of the loss w.r.t. the output of the layer

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. the input of the layer

        """
        batch_size, out_height, out_width, in_channels = upstream_grad.shape
        kernel_height, kernel_width = self.kernel_size
        in_height = (out_height - 1) * self.stride[0] + kernel_height
        in_width = (out_width - 1) * self.stride[1] + kernel_width

        # Compute gradients with respect to input
        input_grad = np.zeros((batch_size, in_height, in_width, in_channels))
        for i in range(out_height):
            for j in range(out_width):
                input_grad[:, i * self.stride[0]:i * self.stride[0] + kernel_height, j * self.stride[1]:j * self.stride[1] + kernel_width, :] = upstream_grad[:, i:i + 1, j:j + 1, :]

        return input_grad


class SoftmaxLayer:
    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        """
        Forward pass of the layer
        Parameters
        ----------
        X : np.array
            Input to the layer of shape (batch_size, n_classes)

        Returns
        -------
        np.array
            Output of the layer of shape (batch_size, n_classes)

        """
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self._prev_result = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self._prev_result

    def backward(self, upstream_grad):
        """
        Backward pass of the layer
        Parameters
        ----------
        upstream_grad : np.ndarray
            Gradient of the loss w.r.t. the output of the layer

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. the input of the layer

        """
        return self._prev_result * (upstream_grad - np.sum(self._prev_result * upstream_grad, axis=1, keepdims=True))


class CNNClassifier(BaseClassifier):
    """
    Convolutional Neural Network Classifier

    Attributes:
        layers: Sequence of layers
        lr: Learning rate
        epochs: Number of epochs
        random_state: Random state
        alpha: Regularization parameter
        batch_size: Batch size
        out_layer_: Output layer
        softmax_layer_: Softmax layer
    """

    def __init__(self, layers: Sequence[Union[ConvLayer, PoolingLayer]], lr: float = 0.01, epochs: int = 1000,
                 random_state: Optional[int] = None, alpha: float = 0.0001, batch_size: int = 32):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.rng_ = np.random.RandomState(random_state)
        self.out_layer_: Optional[ModularLinearLayer] = None
        self.softmax_layer_: Optional[SoftmaxLayer] = None
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the CNN classifier to the training data.

        Parameters
        ----------
        X : np.array
            Input features of shape (n_samples, n_features)
        y : np.array
            Target values of shape (n_samples,)
        """
        n_samples = X.shape[0]
        n_classes = np.max(y) + 1

        # Initialize output layer and softmax layer
        self.out_layer_ = ModularLinearLayer(self.layers[-1].out_channels, n_classes)
        self.softmax_layer_ = SoftmaxLayer()

        # Create a modular MLP with convolutional and pooling layers followed by the output layer
        mlp = [layer for layer in self.layers] + [self.out_layer_, self.softmax_layer_]

        for _ in range(self.epochs):
            # Shuffle the training data
            indices = self.rng_.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                output = X_batch
                for layer in mlp:
                    output = layer(output)

                # Compute loss and accuracy
                loss = self.out_layer_.loss(output, y_batch)
                accuracy = self.out_layer_.accuracy(output, y_batch)

                # Backward pass
                grad = self.out_layer_.grad(output, y_batch)
                for layer in reversed(mlp):
                    grad = layer.backward(grad, alpha=self.alpha)

                # Update parameters
                for layer in mlp:
                    if isinstance(layer, ConvLayer):
                        layer.update(self.lr)

    def predict(self, X: np.array) -> np.array:
        """
        Make predictions using the trained CNN classifier.

        Parameters
        ----------
        X : np.array
            Input features of shape (n_samples, n_features)

        Returns
        -------
        np.array
            Predicted target values of shape (n_samples,)
        """
        output = X
        for layer in self.layers:
            output = layer(output)

        output = self.out_layer_(output)
        predictions = np.argmax(output, axis=1)
        return predictions

    def score(self, X: np.array, y: np.array) -> float:
        """
        Calculate the accuracy of the CNN classifier on the given test data.

        Parameters
        ----------
        X : np.array
            Input features of shape (n_samples, n_features)
        y : np.array
            Target values of shape (n_samples,)

        Returns
        -------
        float
            Accuracy of the classifier on the test data
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
