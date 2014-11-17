import abc


class Feature:
    def __init__(self):
        pass

    @staticmethod
    def discover_features(data):
        """
        Used to discover features automatically from a data matrix.
        :param data: np.ndarray
        :return: Feature[]
        """
        return []

    @abc.abstractmethod
    def extract(self, data):
        """
        Extracts the desired feature from the matrix data. Returns it as a vector.
        :param data: np.ndarray
        :return: np.ndarray
        """
        return 0.

    @abc.abstractmethod
    def get_classifiers(self):
        """
        Offers a list of potential classifiers for this feature (used for seeding an unconfigured algorithm).
        :return: classifier[]
        """
        return []

    @abc.abstractmethod
    def describe(self):
        return ''
