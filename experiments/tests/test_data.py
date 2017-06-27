import os
import unittest

from theano import tensor as T
from theano import function

import numpy as np

from sklearn.metrics import mean_squared_error

from experiments.metrics import brier_score
from experiments.metrics import brier_loss
from experiments.metrics import w_brier_loss

from experiments.data import load_toy_example
from experiments.data import load_blobs
from experiments.data import load_webs
from experiments.data import load_weak_iris


def nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def test_data_consistency(X_t, Z_t, z_t, X_v, Z_v, z_v, Y_v, y_v):
    n_training = X_t.shape[0]
    n_features = X_t.shape[1]
    n_classes = Z_t.shape[1]
    n_validation = X_v.shape[0]

    for matrix in [X_t, Z_t, z_t]:
        assert(matrix.shape[0] == n_training)

    for matrix in [X_t, X_v]:
        assert(matrix.shape[1] == n_features)

    for matrix in [X_v, Z_v, z_v, Y_v, y_v]:
        assert(matrix.shape[0] == n_validation)

    for matrix in [Z_t, Z_v, Y_v]:
        assert(matrix.shape[1] == n_classes)

    for vector in [z_t, z_v, y_v]:
        assert(len(vector.shape) == 1)

    for vector in [y_v]:
        assert(vector.max() < n_classes)

    for vector in [z_t, z_v]:
        assert(vector.max() < 2**n_classes)


class TestData(unittest.TestCase):
    def test_load_toy_example(self):
        training, validation, classes = load_toy_example()
        X_t, Z_t, z_t = training
        X_v, Z_v, z_v, Y_v, y_v = validation
        test_data_consistency(X_t, Z_t, z_t, X_v, Z_v, z_v, Y_v, y_v)

    def test_load_blobs(self):
        training, validation, classes = load_blobs()
        X_t, Z_t, z_t = training
        X_v, Z_v, z_v, Y_v, y_v = validation
        test_data_consistency(X_t, Z_t, z_t, X_v, Z_v, z_v, Y_v, y_v)

    def test_load_weak_iris(self):
        training, validation, classes = load_weak_iris()
        X_t, Z_t, z_t = training
        X_v, Z_v, z_v, Y_v, y_v = validation
        test_data_consistency(X_t, Z_t, z_t, X_v, Z_v, z_v, Y_v, y_v)

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"]== "true", "Skipping this test on Travis CI.")
    def test_load_webs(self):
        training, validation, classes = load_webs()
        X_t, Z_t, z_t = training
        X_v, Z_v, z_v, Y_v, y_v = validation
        test_data_consistency(X_t, Z_t, z_t, X_v, Z_v, z_v, Y_v, y_v)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
