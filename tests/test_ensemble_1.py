import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from src.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
random_forest = RandomForestClassifier()
gradient_boosting = GradientBoostingClassifier()

# Fit models
random_forest.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)

# Make predictions
random_forest_predictions = random_forest.predict(X_test)
gradient_boosting_predictions = gradient_boosting.predict(X_test)

# Evaluate accuracy
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_predictions)

# Print results
print("Random Forest Classifier accuracy on test data:", random_forest_accuracy)
print("Gradient Boosting Classifier accuracy on test data:", gradient_boosting_accuracy)
