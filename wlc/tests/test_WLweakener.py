import unittest

import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from wlc.WLweakener import computeM
from wlc.WLweakener import computeVirtual
from wlc.WLweakener import generateWeak
from wlc.WLweakener import binarizeWeakLabels

from sklearn.preprocessing import label_binarize


class TestWLweakener(unittest.TestCase):

    def test_computeM(self):
        M = computeM(c=3, method='supervised')
        expected = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
        assert_array_equal(M, expected)

        M = computeM(c=3, method='noisy', alpha=0.8)
        expected = np.array([[.8, .1, .1],
                             [.1, .8, .1],
                             [.1, .1, .8]])
        assert_array_almost_equal(M, expected)

        M = computeM(c=4, method='noisy', alpha=0.1)
        expected = np.array([[.1, .3, .3, .3],
                             [.3, .1, .3, .3],
                             [.3, .3, .1, .3],
                             [.3, .3, .3, .1]])
        assert_array_almost_equal(M, expected)

        # FIXME see the reason of the ordering
        M = computeM(c=2, method='quasi_IPL', beta=0.2)
        expected = np.array([[0., 0.],
                             [0., 1.],
                             [1., 0.],
                             [0., 0.]])
        assert_array_equal(M, expected)

        M = computeM(c=3, method='quasi_IPL', beta=0.0)
        expected = np.array([[0., 0., 0.],
                             [0., 0., 1.],
                             [0., 1., 0.],
                             [0., 0., 0.],
                             [1., 0., 0.],
                             [0., 0., 0.],
                             [0., 0., 0.],
                             [0., 0., 0.]])
        assert_array_equal(M, expected)

    def test_computeVirtual(self):
        z = np.array([0, 1, 2, 3])
        z_bin = computeVirtual(z, c=2, method='IPL')
        expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        assert_array_equal(z_bin, expected)

        z_bin = computeVirtual(z, c=2, method='quasi_IPL')
        expected = np.array([[.5, .5], [0, 1], [1, 0], [np.nan, np.nan]])
        assert_array_almost_equal(z_bin, expected)

    def test_generateWeak(self):
        c = 4
        y = np.array([0, 1, 2, 3])
        M = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        z = generateWeak(y, M)
        expected = np.array([8, 4, 2, 1])
        assert_equal(z, expected)

    def test_binarizeWeakLabels(self):
        c = 4
        z = np.array([8, 4, 2, 1])
        z_bin = binarizeWeakLabels(z, c)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                            ])
        assert_equal(z_bin, expected)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
