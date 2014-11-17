import abc


class Feature:
    def __init__(self):
        pass

    @staticmethod
    def discover_features(data):
        return []

    @abc.abstractmethod
    def extract(self, data):
        return 0.

    @abc.abstractmethod
    def get_classifiers(self):
        return []

    @abc.abstractmethod
    def describe(self):
        return ''
