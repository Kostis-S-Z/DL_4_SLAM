"""
Basic Logistic Regression Model used as baseline from Duolingo
"""

from collections import defaultdict, namedtuple
import math
from random import shuffle, uniform

# Sigma is the L2 prior variance, regularizing the baseline model. Smaller sigma means more regularization.
_DEFAULT_SIGMA = 20.0

# Eta is the learning rate/step size for SGD. Larger means larger step size.
_DEFAULT_ETA = 0.1


class LogisticRegressionInstance(namedtuple('Instance', ['features', 'label', 'name'])):
    """
    A named tuple for packaging together the instance features, label, and name.
    """
    def __new__(cls, features, label, name):
        if label:
            if not isinstance(label, (int, float)):
                raise TypeError('LogisticRegressionInstance label must be a number.')
            label = float(label)
        if not isinstance(features, dict):
            raise TypeError('LogisticRegressionInstance features must be a dict.')
        return super(LogisticRegressionInstance, cls).__new__(cls, features, label, name)


class LogisticRegression(object):
    """
    An L2-regularized logistic regression object trained using stochastic gradient descent.
    """

    def __init__(self, sigma=_DEFAULT_SIGMA, eta=_DEFAULT_ETA):
        super(LogisticRegression, self).__init__()
        self.sigma = sigma  # L2 prior variance
        self.eta = eta  # initial learning rate
        self.weights = defaultdict(lambda: uniform(-1.0, 1.0)) # weights initialize to random numbers
        self.fcounts = None # this forces smaller steps for things we've seen often before

    def predict_instance(self, instance):
        """
        This computes the logistic function of the dot product of the instance features and the weights.
        We truncate predictions at ~10^(-7) and ~1 - 10^(-7).
        """
        a = min(17., max(-17., sum([float(self.weights[k]) * instance.features[k] for k in instance.features])))
        return 1. / (1. + math.exp(-a))

    def error(self, instance):
        return instance.label - self.predict_instance(instance)

    def reset(self):
        self.fcounts = defaultdict(int)

    def training_update(self, instance):
        if self.fcounts is None:
            self.reset()
        err = self.error(instance)
        for k in instance.features:
            rate = self.eta / math.sqrt(1 + self.fcounts[k])
            # L2 regularization update
            if k != 'bias':
                self.weights[k] -= rate * self.weights[k] / self.sigma ** 2
            # error update
            self.weights[k] += rate * err * instance.features[k]
            # increment feature count for learning rate
            self.fcounts[k] += 1

    def train(self, train_set, iterations=10):
        for it in range(iterations):
            print('Training iteration ' + str(it+1) + '/' + str(iterations) + '...')
            shuffle(train_set)
            for instance in train_set:
                self.training_update(instance)
        print('\n')

    def predict_test_set(self, test_set):
        return {instance.name: self.predict_instance(instance) for instance in test_set}

