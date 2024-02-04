import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from src.cnn import ConvLayer, PoolingLayer, SoftmaxLayer, CNNClassifier


def main():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize the input data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert the target variable to NumPy arrays
    y_train = LabelEncoder().fit_transform(y_train)
    y_test = LabelEncoder().fit_transform(y_test)

    # Define the layers for the CNN classifier
    layers = [
        ConvLayer(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1),
        PoolingLayer(kernel_size=(2, 2), stride=(2, 2)),
        ConvLayer(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
        PoolingLayer(kernel_size=(2, 2), stride=(2, 2)),
        SoftmaxLayer()
    ]

    # Create an instance of the CNN classifier
    cnn = CNNClassifier(layers=layers, lr=0.01, epochs=10, random_state=42)

    # Fit the classifier to the training data
    cnn.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = cnn.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))


if __name__ == "__main__":
    main()
