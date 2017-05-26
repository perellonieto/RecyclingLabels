import unittest

from theano import tensor as T
from theano import function

import numpy as np
from numpy.testing import assert_equal

from experiments.utils import binarize_weak_labels
from experiments.utils import brier_score
from experiments.utils import w_brier_loss


def nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


class TestUtils(unittest.TestCase):
    def test_binarize_weak_labels(self):
        c = 4
        z = np.array([8, 4, 2, 1])
        z_bin = binarize_weak_labels(z, c)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        assert_equal(z_bin, expected)

    def test_brier_score(self):
        def expanded_bs_v1(p, y, w):
            N, C = p.shape
            bs = 0
            for n in range(N):
                for c in range(C):
                    bs += w[c]*(p[n, c] - y[n, c])**2
            return bs/N

        def expanded_bs_v2(p, y, w):
            N, C = p.shape
            bs = 0
            for c in range(C):
                bs_c = 0
                for n in range(N):
                    bs_c += (p[n, c] - y[n, c])**2
                bs += w[c]*bs_c
            return bs/N

        def expanded_bs_per_class(p, y, w):
            N, C = p.shape
            bs_per_class = np.zeros(C)
            for c in range(C):
                for n in range(N):
                    bs_per_class[c] += (p[n, c] - y[n, c])**2
                bs_per_class[c] *= w[c]/N
            return bs_per_class

        p_tuple = (np.array([[0, 1], [.6, .4], [.7, .3]]),
                   np.array([[0, 1], [.6, .4], [.7, .3]]),
                   np.random.rand(200, 50))
        y_tuple = (np.array([[0, 1], [.6, .4], [.7, .3]]),
                   np.array([[.3, .7], [.5, .5], [.1, .9]]),
                   np.random.rand(200, 50))
        w_tuple = (np.array([.4, .6]),
                   np.array([.9, .1]),
                   np.random.rand(50, 1))

        for p, y, w in zip(p_tuple, y_tuple, w_tuple):
            self.assertAlmostEqual(expanded_bs_v1(p, y, w),
                                   expanded_bs_v2(p, y, w))
            self.assertAlmostEqual(expanded_bs_v2(p, y, w),
                                   brier_score(p, y, w))
            self.assertAlmostEqual(brier_score(p, y, w),
                                   brier_score(p, y, w, per_class=True).sum())
            self.assertTrue(np.allclose(expanded_bs_per_class(p, y, w),
                            brier_score(p, y, w, per_class=True)))
            self.assertEqual(brier_score(p, y, w, per_class=True).shape[0],
                             len(w))

    def test_w_brier_loss(self):
        p_tuple = (np.array([[0, 1], [.6, .4], [.7, .3]]),
                   np.array([[0, 1], [.6, .4], [.7, .3]]),
                   np.random.rand(200, 50))
        y_tuple = (np.array([[0, 1], [.6, .4], [.7, .3]]),
                   np.array([[.3, .7], [.5, .5], [.1, .9]]),
                   np.random.rand(200, 50))
        w_tuple = (np.array([.4, .6]),
                   np.array([.9, .1]),
                   np.random.rand(50))

        for p, y, w in zip(p_tuple, y_tuple, w_tuple):
            o = T.dmatrix('o')
            f = T.dmatrix('f')

            brier_loss = w_brier_loss(o, f, w)
            f_brier_loss = function([o, f], brier_loss)

            self.assertAlmostEqual(brier_score(p, y, w),
                                   f_brier_loss(p, y))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
