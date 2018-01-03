import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from wlc.WLweakener import computeM
from wlc.WLweakener import generateWeak
from wlc.WLweakener import binarizeWeakLabels

# Necessary for the make_blobs modification
# from sklearn.datasets import make_blobs
import numbers
from sklearn.utils import check_array, check_random_state


def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    """Modified version of make_blobs from sklearn.datasets that supports
    n_samples to be an array with the number of samples for each cluster.

    Generate isotropic Gaussian blobs for clustering.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int or array of size centers, optional (default=100)
        The total number of points equally divided among clusters.
    n_features : int, optional (default=2)
        The number of features for each sample.
    centers : int or array of shape [n_centers, n_features], optional
        (default=3)
        The number of centers to generate, or the fixed center locations.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    shuffle : boolean, optional (default=True)
        Shuffle the samples.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    Examples
    --------
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
    See also
    --------
    make_classification: a more intricate variant
    """
    generator = check_random_state(random_state)

    if isinstance(centers, numbers.Integral):
        centers = generator.uniform(center_box[0], center_box[1],
                                    size=(centers, n_features))
    else:
        centers = check_array(centers)
        n_features = centers.shape[1]

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []

    n_centers = centers.shape[0]
    if not isinstance(n_samples, int):
        if len(n_samples) != len(centers):
            raise ValueError("If n_samples is an array, it needs to be of length = centers")
        n_samples_per_center = n_samples
        n_samples = sum(n_samples_per_center)
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1


    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i] + generator.normal(scale=std,
                                               size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        indices = np.arange(n_samples)
        generator.shuffle(indices)
        X = X[indices]
        y = y[indices]

    return X, y

def make_weak_true_partition(M, X, y, true_size=0.1, random_state=None):
    n_c = len(np.unique(y))
    classes = range(0, n_c)
    assert(n_c == np.max(y)+1)
    Y = label_binarize(y, classes)
    z = generateWeak(y, M, seed=random_state)
    Z = binarizeWeakLabels(z, c=n_c)

    # Create partition for weak and true labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=true_size,
                                 random_state=random_state)
    for weak_fold, true_fold in sss.split(X, y):
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


def load_webs(random_state=None, standardize=True):
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

    X = np.load('data/X_full.npy', encoding='latin1')
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
                                        random_state=random_state)
    X_val, Z_val, z_val, Y_val, y_val = shuffle(X_val, Z_val, z_val, Y_val,
                                                y_val,
                                                random_state=random_state)

    # TODO is it possible to keep the matrices sparse with Keras?
    if sparse.issparse(X_train):
        X_train = X_train.toarray()
        Z_train = Z_train
        z_train = z_train
        X_val = X_val.toarray()

    if standardize:
        X_train = scale(X_train)
        X_val = scale(X_val)

    return (X_train, Z_train, z_train), (X_val, Z_val, z_val, Y_val, y_val), categories


def load_toy_example(random_state=None):
    return load_blobs(n_classes=2, random_state=random_state)


def load_blobs(n_samples=1000, n_features=2, n_classes=6, random_state=None):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=n_classes, random_state=random_state)
    if n_classes == 2:
        Y = np.vstack([1-y, y]).T
    else:
        Y = label_binarize(y, range(n_classes))
    # FIXME z should not be exactly y
    Z = Y

    p2 = np.array([2**n for n in reversed(range(n_classes))])
    z = Z.dot(p2)

    X_train, X_val, Z_train, Z_val, z_train, z_val, Y_train, Y_val, y_train, y_val = train_test_split(X, Z, z, Y, y, test_size=0.5, random_state=random_state)

    return (X_train, Z_train, z_train), (X_val, Z_val, z_val, Y_val, y_val), None


def load_weak_blobs(method='quasi_IPL', n_samples=2000, n_features=2,
                    n_classes=6, random_state=None, true_size=0.1, M=None,
                    **kwargs):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=n_classes, random_state=random_state, **kwargs)

    if M is None:
        M = computeM(n_classes, method=method, alpha=0.5, beta=0.3,
                     seed=random_state)
    return make_weak_true_partition(M, X, y, true_size=true_size,
                                    random_state=random_state)


def load_weak_iris(method='quasi_IPL', true_size=0.1, random_state=None):
    # Load original clean data
    iris = load_iris()
    X = iris.data
    y = iris.target
    M = computeM(len(np.unique(y)), method=method, alpha=0.5, beta=0.3,
                 seed=random_state)
    return make_weak_true_partition(M, X, y, true_size=true_size,
                                    random_state=random_state)
