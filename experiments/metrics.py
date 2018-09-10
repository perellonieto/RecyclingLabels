import numpy as np

import theano.tensor as T

from sklearn.metrics import log_loss


def compute_error_matrix(p, err_func):
    error = np.zeros((len(p), len(p)))
    for i, p_prior in enumerate(p):
        for j, p_pred in enumerate(p):
            error[i,j] = err_func(p_pred, float(i==j))
    return error


def compute_expected_error(p, e):
    return np.sum(p*e.sum(axis=1))


def brier_score(target, predicted, class_weights, per_class=False):
    """Brier score between target and predicted

    Parameters
    ----------
    target : numpy.ndarray
        Ground truth of shape (N,C) where N is the number
        of samples and C is the number of classes

    predicted : numpy.ndarray
        Predictions of shape (N,C) where N is the number
        of samples and C is the number of classes

    class_weights : numpy.ndarray, optional
        Array of weights of shape (C,) indicating the importance
        of each class prediction

    per_class : bool, optional
        If true, the error per class is returned in a `ndarray` of size 1xC

    Returns
    -------
    bs : float
        The mean weighted brier score for all the samples

    bs_pc : ndarray, optional
        Only returned if `per_class` is True.
        The weighted mean of all the samples per class with shape (C,)
    """
    if per_class:
        return np.squeeze(np.multiply(
                   np.square(target - predicted).mean(axis=0).reshape(-1, 1),
                   class_weights.reshape(-1, 1)))

    return np.square(target - predicted).dot(class_weights).mean()


def brier_loss(y_true, y_pred):
    """ Computes weighted brier score for the given tensors

    equivalent to:
            w = class_weigths
            N, C = y_true.shape
            bs = 0
            for n in range(N):
                for c in range(C):
                    bs += (y_pred[n, c] - y_true[n, c])**2
            return bs/N
    """
    return T.mean(T.sum(T.square(T.sub(y_pred, y_true)), axis=1))


def w_brier_loss(y_true, y_pred, class_weights):
    """ Computes weighted brier score for the given tensors

    equivalent to:
            w = class_weigths
            N, C = y_true.shape
            bs = 0
            for n in range(N):
                for c in range(C):
                    bs += w[c]*(y_pred[n, c] - y_true[n, c])**2
            return bs/N
    """
    return T.mean(T.dot(T.square(T.sub(y_pred, y_true)), class_weights),
                  axis=-1)

def log_loss(y_true, y_pred):
    return T.mean(T.sum(-y_true*T.log(y_pred), axis=1))
