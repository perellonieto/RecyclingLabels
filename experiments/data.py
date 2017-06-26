import pandas as pd
import numpy as np
from numpy.random import RandomState

from sklearn.utils import shuffle
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from wlc.WLweakener import computeM
from wlc.WLweakener import generateWeak
from wlc.WLweakener import binarizeWeakLabels


def load_webs(seed=None):
    categories = ['blog', 'inmo', 'parking', 'b2c', 'no_b2c', 'Other']
    n_cat = len(categories)

    # Note that the pickle file contains a multi-index dataframe. We take the
    # label (multi-)column only.
    dfY = pd.read_csv('data/Y_newlabels.csv', index_col=0)
    Y_val = dfY[categories].as_matrix()
    y_val = Y_val.argmax(axis=1)

    if np.any(np.sum(Y_val, axis=1) != 1):
        print("Oops!, there are unexpected weak labels in the new dataset."
              "This may cause some errors below")

    X = np.load('data/X_full.npy')
    X = X.item()

    # Label dataframe
    dfZ = pd.read_csv('data/Z_full.csv', index_col=0)

    indices_val = [dfZ.index.get_loc(label) for label in dfY.index]
    Z_val = dfZ.iloc[indices_val][categories].as_matrix().astype(int)
    X_val = X[indices_val, :]

    mask_train = np.ones(len(dfZ), dtype=bool)
    mask_train[indices_val] = False
    Z_train = dfZ[mask_train][categories].as_matrix().astype(int)
    X_train = X[mask_train]

    # Convert weak label vector to integer
    # A useful vector of exponencially decreasing powers of 2.
    p2 = np.array([2**n for n in reversed(range(n_cat))])
    z_train = Z_train.dot(p2)
    z_val = Z_val.dot(p2)

    X_train, Z_train, z_train = shuffle(X_train, Z_train, z_train,
                                        random_state=seed)
    X_val, Z_val, z_val, Y_val, y_val = shuffle(X_val, Z_val, z_val, Y_val,
                                                y_val)

    return (X_train, Z_train, z_train), (X_val, Z_val, z_val, Y_val, y_val), categories


def load_toy_example(seed=None):
    return load_blobs(n_classes=2, seed=seed)


def load_blobs(n_samples=1000, n_features=2, n_classes=6, seed=None):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=n_classes, random_state=seed)
    if n_classes == 2:
        Y = np.vstack([1-y, y]).T
    else:
        Y = label_binarize(y, range(n_classes))
    # FIXME z should not be exactly y
    Z = Y

    p2 = np.array([2**n for n in reversed(range(n_classes))])
    z = Z.dot(p2)

    X_train, X_val, Z_train, Z_val, z_train, z_val, Y_train, Y_val, y_train, y_val = train_test_split(X, Z, z, Y, y, test_size=0.5, random_state=seed)

    return (X_train, Z_train, z_train), (X_val, Z_val, z_val, Y_val, y_val), None


def make_weak_true_partition(M, X, y, true_size=0.1, random_state=0):
    n_c = len(np.unique(y))
    classes = range(0,n_c)
    assert(n_c == np.max(y)+1)
    Y = label_binarize(y, classes)
    z = generateWeak(y, M)
    Z = binarizeWeakLabels(z, c=n_c)

    # Create partition for weak and true labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=true_size,
                                 random_state=random_state)
    weak_fold, true_fold = sss.split(X, y).next()

    # Weaken the true labels fold using the mixing matrix M
    X_w = X[weak_fold]
    z_w = z[weak_fold]
    Z_w = Z[weak_fold]

    # Select the true labels fold
    X_t = X[true_fold]
    z_t = z[true_fold]
    Z_t = Z[true_fold]
    y_t = y[true_fold]
    Y_t = Y[true_fold]

    # TODO refactor name convention of train and val, for weak and true
    return (X_w, Z_w, z_w), (X_t, Z_t, z_t, Y_t, y_t), classes


def load_weak_iris(method='quasi_IPL', true_size=0.1, random_state=0):
    # Load original clean data
    iris = load_iris()
    X = iris.data
    y = iris.target
    M = computeM(len(np.unique(y)), method=method)
    return make_weak_true_partition(M, X, y, true_size=true_size,
                                    random_state=random_state)
