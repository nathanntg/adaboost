from feature import Feature
from adaboost.classifiers import Linear
import numpy as np


class Simple(Feature):
    """
    A Simple feature simply extracts a single column from the data matrix.
    """
    def __init__(self, column):
        Feature.__init__(self)
        self.column = column

    @staticmethod
    def discover_features(data):
        return [Simple(i) for i in xrange(0, data.shape[1])]

    def extract(self, data):
        return data[:, self.column]

    def get_classifiers(self):
        return [Linear(self)]

    def describe(self):
        return 'Column ' + str(self.column)

