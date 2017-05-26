import numpy as np

import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from functools import partial


def binarize_weak_labels(z, c):
    """
    Binarizes the weak labels depending on the method used to generate the weak
    labels.

    Args:
        z       :List of weak labels. Each weak label is an integer whose
                 binary representation encondes the observed weak labels
        c       :Number of classes. All components of z must be smaller than
                 2**c
    Returns:
        z_bin
    """
    # Transform the weak label indices in z into binary label vectors
    z_bin = np.zeros((z.size, c), dtype=int)       # weak labels (binary)
    for index, i in enumerate(z):         # From dec to bin
        z_bin[index, :] = [int(x) for x in np.binary_repr(i, width=c)]

    return z_bin


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


def create_model(input_dim=1, output_size=1, optimizer='rmsprop',
                 init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                 nesterov=False, loss='mean_squared_error',
                 class_weights=None):
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, kernel_initializer=init,
                    activation='sigmoid'))
    model.add(Dense(output_size, input_dim=20, kernel_initializer=init,
                    activation='softmax'))

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay,
                        nesterov=nesterov)

    if class_weights is None:
        class_weights = np.ones(output_size)

    if loss == 'w_brier_score':
        loss = partial(w_brier_loss, class_weights=class_weights)
        loss.__name__ = 'w_brier_score'
    elif loss == 'brier_score':
        loss = brier_loss
        loss.__name__ = 'brier_score'

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['acc'])
    return model


def w_brier_loss(y_true, y_pred, class_weights):
    """

    class_weights: array with (c,) dimensions
    """
    return T.mean(T.dot(T.square(T.sub(y_pred, y_true)), class_weights),
                  axis=-1)


def uniqueness(df):
    unique_indices = df.index.unique().shape[0] == df.shape[0]
    if not unique_indices:
        where = np.where((itemfreq(df.index.astype('int'))[:, 1] > 1))[0]
    return unique_indices


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

