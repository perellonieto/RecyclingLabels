import unittest

import numpy as np
from numpy.testing import assert_equal

from experiments.utils import binarize_weak_labels


class TestUtils(unittest.TestCase):
    def test_binarize_weak_labels(self):
        c = 4
        z = np.array([8, 4, 2, 1, 2, 8])
        z_bin = binarize_weak_labels(z, c)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [1, 0, 0, 0]])
        assert_equal(z_bin, expected)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
