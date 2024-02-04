import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional
import numpy as np

from src.base import BaseClassifier
from src.tree import DecisionTree  # Import the DecisionTree from previous session (Optional)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.preprocessing import LabelEncoder

class RandomForestClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, max_features: str = 'sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.estimators_ = []
        self.classes_ = None
        self.feature_sets_ = []  # Keep track of the features used by each tree
    
    def fit(self, X, y):
        # First we find out the unique classes in the target labels
        self.classes_ = np.unique(y)
        n_features = X.shape[1]   # Calculate number of features
        
        # Choose max number of features to use in each Decision Tree 
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features

        # Create decision trees based on the number of estimators
        for _ in range(self.n_estimators):
            # Bootstrap sample (random sampling with replacement)
            sample_idxs = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            X_sample, y_sample = X[sample_idxs], y[sample_idxs] 
            
            # Select random subset of features for each tree
            feature_idxs = np.random.choice(np.arange(n_features), size=max_features, replace=False)
            self.feature_sets_.append(feature_idxs)
            X_sample_sub = X_sample[:, feature_idxs]

            # Create and fit decision tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample_sub, y_sample) # Fitting using Bootstrap sample and subset of features
            self.estimators_.append(tree)

    def predict(self, X):
        # Each decision tree makes a prediction based on subset of features
        predictions = np.array([tree.predict(X[:, feature_idxs]) for tree, feature_idxs in zip(self.estimators_, self.feature_sets_)])
        predictions = predictions.astype(int)  # Convert predictions to integers
        # Return most common prediction
        return np.array([np.argmax(np.bincount(pred)) for pred in predictions.T])

    def score(self, X, y):
        return super().score(X, y)


class GradientBoostingClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.clf = SklearnGBC(n_estimators=self.n_estimators,
                              learning_rate=self.learning_rate,
                              random_state=self.random_state)
        self.classes_ = None
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y):
        # First we find out the unique classes in the target labels
        self.classes_ = np.unique(y)
        # Encode the target labels to integers which is a common practice for gradient boosting algorithms
        y_encoded = self.label_encoder_.fit_transform(y)
        self.clf.fit(X, y_encoded) # Fit the SklearnGBC

    def predict(self, X):
        encoded_predictions = self.clf.predict(X) # Predict classes using SklearnGBC
        # Inverse of encoding transformation to give predictions in the original form of target labels
        return self.label_encoder_.inverse_transform(encoded_predictions)
    
    def score(self, X, y):
        return super().score(X, y)
