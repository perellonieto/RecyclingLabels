import numpy as np


from scipy import sparse

import theano.tensor as T
import matplotlib.pyplot as plt
from matplotlib import cm

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam

from functools import partial

seed = 42


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

def newfig(name):
    fig = plt.figure(name, figsize=(6,4))
    fig.clf()
    return fig


def savefig(fig, path='figures', prefix='weak_labels_', extension='svg'):
    fig.tight_layout()
    name = fig.get_label()
    filename = "{}{}.{}".format(prefix, name, extension)
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    fig.savefig(os.path.join(path, filename))


def plot_data(x, y, loc='best', save=True, title='data', cmap='Paired'):
    if sparse.issparse(x):
        x = x.toarray()

    fig = newfig('data')
    ax = fig.add_subplot(111)

    classes = np.unique(y)
    n_c = float(len(classes))

    cmap = cm.get_cmap(cmap)
    for i, y_i in enumerate(classes):

        ax.scatter(x[y==y_i, 0], x[y==y_i, 1], c=cmap(i/n_c), s=30,
                   edgecolors=None, cmap=cmap, alpha=.8, lw=0.1,
                   label='{}'.format(y_i))
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_title(title)
    ax.axis('equal')
    ax.legend() #loc=loc)
    ax.grid(True)
    if save:
        savefig(fig)
    return fig
