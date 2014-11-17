import abc

class Classifier:
    def __init__(self, feature):
        self.feature = feature

    @abc.abstractmethod
    def describe(self):
        return ''
