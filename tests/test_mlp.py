import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.mlp import MLPRegressor

# Load the Boston Housing dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the MLPRegressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', lr=0.0001, epochs=50, random_state=42)

# Fit the MLPRegressor to the training data
mlp_regressor.fit(X_train_scaled, y_train)

# Evaluate the MLPRegressor on the test data
score = mlp_regressor.score(X_test_scaled, y_test)
print("MLPRegressor R2 score on test data:", score)

# Make predictions using the trained MLPRegressor
predictions = mlp_regressor.predict(X_test_scaled)
print("Predicted target values:", predictions)
print("Expected target values:", y_test)
