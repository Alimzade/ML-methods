import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import make_blobs
from src.clustering import KMeans

# Generate synthetic data
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Initialize KMeans model
kmeans = KMeans(n_clusters=3)

# Fit the KMeans model
kmeans.fit(X)

# Get the predicted labels
labels = kmeans.predict(X)

# Print the predicted labels
print("Predicted labels:", labels)
kmeans.score(X)
