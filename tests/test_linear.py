import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# here import the ready data and test on my code 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.linear import LinearRegressor, LogisticRegression, SGDRegression


# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
linear_regressor = LinearRegressor()
logistic_regressor = LogisticRegression()
sgd_regressor = SGDRegression()

# Fit models
linear_regressor.fit(X_train, y_train)
logistic_regressor.fit(X_train, y_train)
#sgd_regressor.fit(X_train, X_test)

# Evaluate models
print("Linear Regression R2 score on test data:", linear_regressor.score(X_test, y_test))
print("Logistic Regression accuracy on test data:", logistic_regressor.score(X_test, y_test))
#print("SGD Regression score on test data:", sgd_regressor.score(X_test, y_test))




# Make predictions
#linear_predictions = linear_regressor.predict(X_test)
#logistic_predictions = logistic_regressor.predict(X_test)

# Evaluate predictions
##print("LinearRegression predictions on test data:", linear_predictions)
#print("LogisticRegressor predictions on test data:", logistic_predictions)

#print("Expected target values:", y_test)

