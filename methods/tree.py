import os
import sys

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional
import numpy as np

from src.base import BaseClassifier


class DecisionNode(BaseClassifier):
    NODEID = 0

    def __init__(self, attr=-1, children=None, label=None):
        super().__init__()
        self.attr = attr
        self.children = children
        self.label = label
        self.id = DecisionNode.NODEID
        # Incrementing the NODEID after each node creation
        DecisionNode.NODEID += 1

class DecisionTree(BaseClassifier):
    """
    Class for decision tree using the ID3 algorithm.
    """

    def __init__(self, criterion: str = 'entropy', random_state: Optional[int] = None):
        self.rng_ = np.random.RandomState(random_state)
        
        # Checking the criterion for impurity and assigning the respective function
        if criterion == 'entropy':
            self.criterion = self._entropy
        elif criterion == 'gini':
            self.criterion = self._gini
        elif criterion == 'misclass':
            self.criterion = self._misclass
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Allowed values are: 'entropy', 'gini', 'misclass'")

        
    def fit(self, X, y, verbose=0):

        # Input validation
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of instances"

        def _fit(X, y, attributes=None):

            # Checking if no data is left
            if len(X) == 0:
                # If there are no instances left, return a leaf node with the most common label in y
                label, _ = self.most_common_class(y)
                return DecisionNode(label=label)
            
            # Set up temporary variables
            N, d = X.shape
            # Checking if no attributes are provided then initializing them with unique values from data
            if attributes is None:
                attributes = {a_i: np.unique(X[:, a_i]) for a_i in range(d)}
            attributes = attributes.copy()  # Make a copy before modification
            depth = d - len(attributes) + 1
            
            #if len(X) == 0: return DecisionNode()
            
            label, fIsPure = self.most_common_class(y)
            # Stop criterion 1: If data is pure, create a leaf node with label
            if fIsPure: 
                if verbose: print ("\t\t Leaf Node with label %s due to purity." % label)
                return DecisionNode(label=label)
            
            # Stop criterion 2: If no attributes left, create a leaf node with label
            if len(attributes) == 0:
                if verbose: print ("\t\t Leaf Node with label %s due to exhausted attributes." % label)
                return DecisionNode(label=label)
            
            # Get attribute with maximum impurity reduction (best attribute for splitting)
            a_i, a_ig = self.get_split_attribute(X, y, attributes, self.criterion, verbose=verbose)
            if verbose: print ("Level %d: Choosing attribute %d out of %s with gain %f" % (depth, a_i, attributes.keys(), a_ig))
            
            values = attributes.pop(a_i)
            splits = [X[:,a_i] == v for v in values]  
            branches = {}
            
            # For each unique value of the selected attribute, split the data
            for v, split in zip(values, splits):
                if not np.any(split):
                    if verbose: print ("Level %d: Empty split for value %s of attribute %d" % (depth, v, a_i))
                    branches[v] = DecisionNode(label=label)
                else: 
                    if verbose: print ("Level %d: Recursion for value %s of attribute %d" % (depth, v, a_i))
                    branches[v] = _fit(X[split,:], y[split], attributes=attributes.copy())
            
            return DecisionNode(attr=a_i, children=branches, label=label)
   
        self.root = _fit(X, y)
        return self


    def predict(self, X):
        def _predict(x, node):
            if not node.children:
                return node.label
            else:
                attr_value = x[node.attr]
                for value, child_node in node.children.items():
                    if isinstance(value, tuple):
                        # Handle threshold-based splits for floating-point attribute values
                        if attr_value <= value[0]:
                            return _predict(x, child_node)
                    else:
                        # Handle regular splits for categorical attribute values
                        if attr_value == value:
                            return _predict(x, child_node)
                # Return the most common class label among training instances that reach this leaf node
                return node.label

        return [_predict(x, self.root) for x in X]
            

    # Impurity functions (gini, entropy, misclass)
    def _gini(self, p):
        """
        p: class frequencies as numpy array with np.sum(p)=1
        returns: impurity according to gini criterion
        """
        return 1. - np.sum(p**2)

    def _entropy(self, p):
        """
        p: class frequencies as numpy array with np.sum(p)=1
        returns: impurity according to entropy criterion
        """
        idx = np.where(p==0.) #consider 0*log(0) as 0
        p[idx] = 1.
        r = p*np.log2(p)
        return -np.sum(r)

    def _misclass(self, p):
        """
        p: class frequencies as numpy array with np.sum(p)=1
        returns: impurity according to misclassification rate
        """
        return 1-np.max(p)


    # Impurity reduction 
    def impurity_reduction(self, X, a_i, y, impurity, verbose=0):
        """
        X: data matrix n rows, d columns
        a_i: column index of the attribute to evaluate the impurity reduction for
        y: concept vector with n rows and 1 column
        impurity: impurity function of the form impurity(p_1....p_k) with k=|X[a].unique|
        returns: impurity reduction
        Note: for more readable code we do not check any assertion 
        """
        N, d = float(X.shape[0]), float(X.shape[1])
        
        y_v = np.unique(y)
        
        # Compute relative frequency of each class in X
        p = (1. / N) * np.array([np.sum(y==c) for c in y_v])
        # ..and corresponding impurity l(D)
        H_p = impurity(p)
        if verbose: print ("\t Impurity %0.3f: %s" % H_p)
        
        a_v = np.unique(X[:, a_i])
        
        # Create and evaluate splitting of X induced by attribute a_i
        # We assume nominal features and perform m-ary splitting
        H_pa = []
        for a_vv in a_v:
            mask_a = X[:, a_i] == a_vv
            N_a = float(mask_a.sum())
            
            # Compute relative frequency of each class in X[mask_a]
            pa = (1. / N_a) * np.array([np.sum(y[mask_a] == c) for c in y_v])
            H_pa.append((N_a / N) * impurity(pa))
            if verbose: print ("\t\t Impurity %0.3f for attribute %d with value %s: " % (H_pa[-1], a_i, a_vv))
            
        IR = H_p - np.sum(H_pa)
        if verbose:  print ("\t Estimated reduction %0.3f" % IR)

        # Return the impurity reduction
        return IR
    

    # Select best attribute for splitting (index with the maximum impurity reduction)
    def get_split_attribute(self, X, y, attributes, impurity, verbose=0):
        """
        X: data matrix n rows, d columns
        y: vector with n rows, 1 column containing the target concept
        attributes: A dictionary mapping an attribute's index to the attribute's domain
        impurity: impurity function of the form impurity(p_1....p_k) with k=|y.unique|
        returns: (1) idx of attribute with maximum impurity reduction and (2) impurity reduction
        """

        N, d = X.shape

        # For each attribute, calculate its impurity reduction
        IR = [0.] * d
        for a_i in attributes.keys():
            IR[a_i] = self.impurity_reduction(X, a_i, y, impurity, verbose)

        # Select attribute with maximum impurity reduction using random choice if there's a tie
        max_ir = max(IR)
        max_indices = [i for i, ir in enumerate(IR) if ir == max_ir]
        b_a_i = self.rng_.choice(max_indices)
        return b_a_i, max_ir

    
    # Determine most frequent class labels 'y', check for purity (further splitting is required or not)
    def most_common_class(self, y):
        """
        :param y: the vector of class labels, i.e. the target
        returns: (1) the most frequent class label in 'y' and (2) a boolean flag indicating whether y is pure
        """
        # Determine the most common class and check if all instances have the same class
        y_v, y_c = np.unique(y, return_counts=True)
        label = y_v[np.argmax(y_c)]
        fIsPure = len(y_v) == 1
        return label, fIsPure

    # Calculate the accuracy of the decision tree on test data
    def score(self, X_test, y_test):
        return super().score(X_test, y_test)

    # Draw the representation of tree structure
    def draw_tree(self):
        def _draw_node(node, level=0):
            prefix = "|   " * level
            if node.label is not None:
                print(f"{prefix}Level {level}: Label: {node.label}")
            else:
                print(f"{prefix}Level {level}: Attribute: {node.attr}")
                for value, child in node.children.items():
                    print(f"{prefix}|-- Value: {value}")
                    _draw_node(child, level + 1)

        _draw_node(self.root, 0)