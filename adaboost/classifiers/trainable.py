from classifier import Classifier
from adaboost import utilities
import numpy as np

class Trainable(Classifier):
    threshold = None
    error = None
    precision = None

    def __init__(self, feature):
        Classifier.__init__(self, feature)

    def get_sort_func(self):
        return None

    def process_data(self, val):
        return val

    def get_potential_boundaries(self, processed_data):
        return range(0, len(processed_data) + 1)

    def to_precision(self, err):
        return abs(err - 0.5)

    def describe(self):
        return self.feature.describe() + ' > ' + str(self.threshold)

    def ready_data(self, data, actual, weights):
        # process data (allow for some sort of pre-processing)
        processed_data = self.process_data(self.feature.extract(data))

        # make paired lists of processed data, actual classification and weights
        lst = [(x, actual[i, 0], weights[i, 0]) for i, x in enumerate(processed_data)]
        lst.sort(cmp=self.get_sort_func(), key=lambda entry: entry[0])

        # figure out initial precision
        best_threshold = lst[0][0] - 1e-5
        err = np.sum(weights[actual == -1])
        best_err = err
        best_precision = self.to_precision(err)
        last_boundary = 0

        # consider all boundaries
        for boundary in self.get_potential_boundaries(processed_data):
            # 0 case taken care of by the initial best_threshold and best_precision
            if 0 == boundary:
                continue

            # advance boundary
            for j in xrange(last_boundary, boundary):
                if 0 < lst[j][1]:
                    err += lst[j][2]
                else:
                    err -= lst[j][2]
            last_boundary = boundary

            # get precision
            precision = self.to_precision(err)

            # is improvement
            if precision > best_precision:
                best_err = err
                best_precision = precision
                if boundary == len(lst):
                    best_threshold = lst[-1][0] + 1e-5
                else:
                    best_threshold = (lst[boundary - 1][0] + lst[boundary][0]) / 2.

        self.threshold = best_threshold
        self.error = best_err
        self.precision = best_precision

    def classify_data(self, data):
        val = self.process_data(self.feature.extract(data))

        # 1 for true, -1 for false
        ret = np.ones((data.shape[0], 1))
        ret[val <= self.threshold] = -1
        return ret