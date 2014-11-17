import features
import classifiers
import copy
import math
import numpy as np


class AdaBoost:
    debug = 0
    features = []
    classifiers = []
    max_iterations = 5
    target_error = 0.01

    # data
    data = None
    actual = None

    # training weights
    weights = None

    # trained parameters
    selected_classifiers = None
    alphas = None

    def __init__(self):
        pass

    def seed_features(self):
        if 0 == len(self.features):
            self.features = features.Simple.discover_features(self.data)

    def seed_classifiers(self):
        if 0 == len(self.classifiers):
            for f in self.features:
                self.classifiers.extend(f.get_classifiers())

    def _evaluate_classifier(self, classifier):
        # get classifiers
        classifier.ready_data(self.data, self.actual, self.weights)

        if 1 < self.debug:
            print 'Eval: ', classifier.describe(), ' (precision: ', classifier.precision, '; error: ', classifier.error, ')'

        return classifier.precision, classifier.error

    def _train_iteration(self):
        # classifier count
        classifier_count = len(self.classifiers)

        # track best
        best_precision = None
        best_error = None
        best_j = None
        for j in xrange(0, classifier_count):
            # get precision
            precision, error = self._evaluate_classifier(self.classifiers[j])

            # best?
            if best_precision is None or precision > best_precision:
                best_precision = precision
                best_error = error
                best_j = j

        # inversion
        if 0.5 < best_error:
            invert = -1
            best_error = 1 - best_error
        else:
            invert = 1

        # new classifier
        classifier = copy.copy(self.classifiers[best_j])
        alpha = 0.5 * invert * np.log((1 - best_error) / best_error)

        # print classifier
        if 0 < self.debug:
            print 'Added: ', classifier.describe(), ' (alpha = ', alpha, ')'

        # add alpha
        self.alphas.append(alpha)
        self.selected_classifiers.append(classifier)

        # get mistakes
        mistakes = (classifier.classify_data(self.data) != self.actual)

        if 0 > invert:
            mistakes = ~mistakes

        sum_of_weights = np.sum(self.weights * mistakes)
        num_actual_0 = np.sum(self.actual[mistakes] == -1)
        num_actual_1 = np.sum(self.actual[mistakes] == 1)

        if 1 < self.debug:
            print 'Incorrect: ', np.sum(mistakes), '; Correct: ', np.sum(~mistakes)
            print 'False positives: ', num_actual_0, '; False negatives: ', num_actual_1

        # adjusters
        self.weights[mistakes] *= 0.5 / sum_of_weights
        self.weights[~mistakes] *= 0.5 / (1 - sum_of_weights)

    def train(self, data, actual):
        # check that data is a matrix (columns = features; rows = entries)
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # check that actual is a column vector
        if not isinstance(actual, np.ndarray):
            actual = np.array(actual)
        if 1 == len(actual.shape):
            actual.shape = [len(actual), 1]
        elif 1 == actual.shape[0]:
            actual = actual.T

        unique_actual = np.unique(actual)
        if 2 < len(unique_actual):
            raise Exception('Actual values must contain only binary classification.')
        if not 1 in unique_actual:
            raise Exception('Must have at least one positive binary classification.')
        if 0 in unique_actual:
            actual[actual == 0] = -1
        elif not -1 in unique_actual:
            raise Exception('Must have at least one negative binary classification.')

        # store
        self.data = data
        self.actual = actual

        # seed configuration
        self.seed_features()
        self.seed_classifiers()

        # fill initial weights
        self.weights = np.ones(actual.shape) / actual.shape[0]

        # initial values
        self.selected_classifiers = []
        self.alphas = []

        # iterations
        for i in xrange(0, self.max_iterations):
            if 1 < self.debug:
                print "Iteration ", i + 1

            # run iteration
            self._train_iteration()

    def apply_to_matrix(self, data):
        # check that data is a matrix (columns = features; rows = entries)
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # build return array
        ret = np.zeros((data.shape[0], 1))

        for i, c in enumerate(self.selected_classifiers):
            ret += c.classify_data(data) * self.alphas[i]

        return ret
