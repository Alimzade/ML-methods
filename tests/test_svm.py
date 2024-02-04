import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.svm import LinearSVM
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the LinearSVM classifier
svm_classifier = LinearSVM(lr=0.001, tol=1e-4, max_iter=1000)

# Fit the LinearSVM classifier to the training data
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the LinearSVM classifier on the test data
accuracy = svm_classifier.score(X_test_scaled, y_test)
print("LinearSVM accuracy on test data:", accuracy)

# Make predictions using the trained LinearSVM classifier
predictions = svm_classifier.predict(X_test_scaled)
print("Predicted class labels:", predictions)
print("True class labels:", y_test)
