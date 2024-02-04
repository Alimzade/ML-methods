import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from src.naive_bayes import MultinomialNaiveBayes
import numpy as np
   
# Load the 20 Newsgroups dataset
categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.baseball', 'sci.space']
data = fetch_20newsgroups(categories=categories)
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the input text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Convert X_train_vectorized and y_train to numpy arrays
X_train_array = X_train_vectorized.toarray()
y_train_array = np.array(y_train)

# Initialize the MultinomialNaiveBayes classifier
nb_classifier = MultinomialNaiveBayes()

# Fit the MultinomialNaiveBayes classifier to the training data
nb_classifier.fit(X_train_array, y_train_array)

# Convert X_test_vectorized to a list of strings
X_test_array = X_test_vectorized.toarray()
X_test_strings = [str(doc) for doc in X_test_array]

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test_strings)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("MultinomialNaiveBayes accuracy on test data:", accuracy)

# Convert X_test_vectorized to a list of strings
X_test_list = vectorizer.inverse_transform(X_test_vectorized)
X_test_strings = [" ".join(doc) for doc in X_test_list]

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test_strings)

# Compare predicted labels with true labels
for pred, true in zip(y_pred, y_test):
    print("Predicted:", pred)
    print("True:", true)
    print("----------------------")