import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.tree import DecisionTree

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Initialize the DecisionTreeClassifier
dt_classifier = DecisionTree()

# Fit the DecisionTreeClassifier to the training data
dt_classifier.fit(X_train, y_train)

# Evaluate the DecisionTreeClassifier on the test data
accuracy = dt_classifier.score(X_test, y_test)
print("DecisionTreeClassifier accuracy on test data:", accuracy)

# Make predictions using the trained DecisionTreeClassifier
predictions = dt_classifier.predict(X_test)
print("Predicted class labels:", predictions)
print("True class labels:", y_test)
