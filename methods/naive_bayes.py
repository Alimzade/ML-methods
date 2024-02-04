import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.base import BaseClassifier


class MultinomialNaiveBayes(BaseClassifier):
    """
     Multinomial Naive Bayes

     Parameters
     -----------
     :param x               : np array
                            input training points, size N:D
     :param y               : np array
                            target/expected values
     :param data_info       : dict
                            0: data label/ y (exptected value)
                            doc_count : number of document that belongs to label 0
                            tokens_count : number of tokens the belongs to label 0
                            tokens : all the tokens and their number of occurences in the label 0

                                Eg:. { 0: {'doc_count' : 480,
                                        'tokens_count' : 14552
                                        'tokens' : {'date' : 33, 'from' : 23}
                                        }
                                    }
     :param total_document_count: int
                            Total number of documents in the whole datasets
     :param vocabulary      : dict
                            dictionary of all the vocab present in the datasets including their number of occurences

                                Eg:. {'date' : 20, 'from' : 23}

     :param priors          : dict
                            {'0' : 0.235, '1' : 0.568}
                            0 : category/lass
                            0.235 : prior probability value

                            Prior probability for each class/category/label

                                Eg:. log(p(c=0)) = log(number of items in c = 0) - log(total items in whole datasets)
                                log(p(c=11)) = log(480) - log(11314)

     :param conditionals    : dict
                            {0 : {'date': 0.356,
                                'from' : 0.557}
                            }
                            Conditional probability of each term in input datasets

                                Eg:. conditional probability of a term = log(term count in particular class) - log(token size in a class + size of vocabulary)
                                p(A/B) = p(A intersection B) / p(B)

     :param alpha           : float, Laplace smoothing parameter (default=1.0)
     """

    def __init__(self, alpha: float = 1.0):
        self.x = []
        self.y = []
        self.data_info = None
        self.total_document_count = None
        self.vocabulary = {}
        self.priors = {}
        self.conditionals = {}
        self.alpha = alpha

    def fit(self, features: np.array, targets: np.array):

        # Error handling for fit
        # Check if features and targets are numpy arrays
        if not isinstance(features, np.ndarray) or not isinstance(targets, np.ndarray):
            raise TypeError('Features and targets must be numpy arrays.')

        # Check if features and targets have correct shapes (same number of samples)
        if features.shape[0] != targets.shape[0]:
            raise ValueError(f'Features and targets should have the same number of samples. {features.shape[0]} samples in features and {targets.shape[0]} samples in targets.')

        self.x = features
        self.y = targets

        # Calculate data_info dictionary
        self.data_info = {}
        unique_labels, label_counts = np.unique(targets, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            self.data_info[label] = {
                'doc_count': count,
                'tokens_count': 0,
                'tokens': {}
            }

        # Count tokens and build vocabulary
        self.total_document_count = features.shape[0]
        for i in range(self.total_document_count):
            label = targets[i]
            tokens = features[i]

            self.data_info[label]['tokens_count'] += len(tokens)
            for token in tokens:
                self.data_info[label]['tokens'][token] = self.data_info[label]['tokens'].get(token, 0) + 1
                self.vocabulary[token] = self.vocabulary.get(token, 0) + 1

        # Calculate priors
        for label, info in self.data_info.items():
            self.priors[label] = np.log(info['doc_count']) - np.log(self.total_document_count)

        # Calculate conditionals
        for label, info in self.data_info.items():
            self.conditionals[label] = {}
            token_size = info['tokens_count']
            for token in self.vocabulary:
                token_count = info['tokens'].get(token, 0)
                self.conditionals[label][token] = np.log(token_count + self.alpha) - np.log(token_size + self.alpha * len(self.vocabulary))

    def predict(self, documents):

        #Error handling for predict
        # Check if documents is a list of strings
        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
            raise TypeError('Documents must be a list of strings.')

        predictions = []

        for doc in documents:
            scores = {}
            tokens = doc.split()  # Tokenize document

            # Calculate the score for each class
            for label, info in self.data_info.items():
                score = self.priors[label] 

                # Update the score based on the tokens in the document
                for token in tokens:
                    if token in self.conditionals[label]:
                        # Add the conditional log-probability of the token
                        score += self.conditionals[label][token]
                    else:
                        # Add the log-probability of an unseen token (using Laplace smoothing)
                        score += np.log(self.alpha) - np.log(info['tokens_count'] + self.alpha * len(self.vocabulary))

                scores[label] = score

            predicted_label = max(scores, key=scores.get) # Choose the class with the highest score as the prediction
            predictions.append(predicted_label) # Add the prediction to the list

        return np.array(predictions)
    
    def score(self, X_test, y_test):
        return super().score(X_test, y_test)