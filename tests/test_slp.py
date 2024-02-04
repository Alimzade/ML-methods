import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.slp import SLPClassifier
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

# Initialize the SLPClassifier
slp_classifier = SLPClassifier(hidden_layer_size=5, lr=0.1, epochs=200, random_state=42)

# Fit the SLPClassifier to the training data
slp_classifier.fit(X_train_scaled, y_train)

# Evaluate the SLPClassifier on the test data
accuracy = slp_classifier.score(X_test_scaled, y_test)
print("SLPClassifier accuracy on test data:", accuracy)

# Make predictions using the trained LinearSVM classifier
predictions = slp_classifier.predict(X_test_scaled)
print("Predicted class labels:", predictions)
print("True class labels:", y_test)