from abc import abstractmethod
import numpy as np


class BaseEstimator:
    @abstractmethod
    def fit(self, X: np.array, y: np.array):
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :param y: numpy array of shape (N, 1) with N being the number of samples as in the provided features and 1 being the number of target dimensions
        :return:
        """
        raise NotImplementedError


class BaseTransformer(BaseEstimator):
    @abstractmethod
    def transform(self, X: np.array) -> np.array:
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :return: numpy array of shape (N, d') with N being the number of samples and d' being the number of feature dimensions after transformation
        """
        raise NotImplementedError


class BaseClassifier(BaseEstimator):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: np array of shape (N, d)
        :return:
        """
        raise NotImplementedError

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :param y: numpy array of shape (N, 1) with N being the number of samples as in the provided features and 1 being the number of target dimensions
        :return: accuracy
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class BaseRegressor(BaseEstimator):
    @abstractmethod
    def predict(self, X: np.ndarray):
        """
        :param X: np array of shape (N, d)
        :return:
        """
        raise NotImplementedError

    def score(self, X: np.ndarray, y:np.ndarray) -> float:
        """
        :param X: numpy array of shape (N, d) with N being the number of samples and d being the number of feature dimensions
        :param y: numpy array of shape (N, 1) with N being the number of samples as in the provided features and 1 being the number of target dimensions
        :return: R2 score
        """
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)
