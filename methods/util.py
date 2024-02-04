from typing import Optional, Tuple

import numpy as np
import pandas as pd


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    :param X:
    :param y:
    :param test_size:
    :param random_state:
    :return:
    """
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(np.arange(X.shape[0]))
    test_size = int(X.shape[0] * test_size)
    test_ind, train_ind = indices[:test_size], indices[test_size:]
    return X[train_ind], X[test_ind], y[train_ind], y[test_ind]


#TODO test this
def normalize(X, upper_bound=1):
    X /= np.max(np.abs(X), axis=0)
    X *= (upper_bound / X.max())
    return X


def make_classification(n_samples: int = 2000, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(random_state)

    # Generate (= sample) artificial algorithm training times
    actual_training_time = rng.uniform(100, 300, n_samples)
    # Add noise to the
    training_time = actual_training_time + rng.normal(0, 10, n_samples)

    actual_power_consumption = rng.normal(1000, actual_training_time)
    power_consumption = actual_power_consumption + rng.normal(0, 50, n_samples)

    actual_probs = normalize(1 + np.sin(actual_training_time + actual_power_consumption))
    cut_probs = np.array(actual_probs > 0.5, dtype=np.int)

    draw_classes = rng.binomial(4, actual_probs)

    data = pd.DataFrame(np.stack([training_time, power_consumption, cut_probs], axis=1),
                        columns=["training_time", "power_consumption", "cut_probs"])
    target = draw_classes
    return data, target


def make_regression(n_samples: int = 2000, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(random_state)

    performance_intercept = 72  #
    performance_slope = 4.9
    training_time = rng.uniform(100, 300, n_samples)
    noisy_training_time = training_time + rng.normal(0, 10, n_samples)
    performances = performance_intercept + performance_slope * training_time + rng.normal(0, training_time / 100)

    def normalize(X, upper_bound=1):
        X /= np.max(np.abs(X), axis=0)
        X *= (upper_bound / X.max())
        return X

    performances = normalize(performances)
    data = pd.DataFrame([noisy_training_time], columns=["training_time"])
    return data, performances
