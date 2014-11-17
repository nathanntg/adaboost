import abc


class Classifier:
    error = None
    precision = None

    def __init__(self, feature):
        self.feature = feature

    @abc.abstractmethod
    def describe(self):
        return ''

    @abc.abstractmethod
    def ready_data(self, data, actual, weights):
        """
        Takes a data matrix, a column vector of actual classifications (-1 or 1) and a column vector of weights and
        customizes the classifier accordingly (e.g., sets the threshold). If this function returns true, it must
        fill in the error and precision parameters based on the data.
        :param data: np.ndarray
        :param actual: np.ndarray
        :param weights: np.ndarray
        :return: bool
        """
        pass

    @abc.abstractmethod
    def classify_data(self, data):
        """
        Takes a data matrix and returns a set of predicted classifications (either -1 or 1).
        :param data: np.ndarray
        :return: np.ndarray
        """
        pass
