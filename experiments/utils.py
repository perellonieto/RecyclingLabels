import numpy as np

import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from functools import partial


def binarizeWeakLabels(z, c):
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
