import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.utils import shuffle
from sklearn.datasets import load_iris, load_digits, make_classification
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from wlc.WLweakener import computeM
from wlc.WLweakener import generateWeak
from wlc.WLweakener import binarizeWeakLabels
from wlc.WLweakener import weak_to_decimal
from wlc.WLweakener import weak_to_index

from sklearn.linear_model import LogisticRegression

# Necessary for the make_blobs modification
# from sklearn.datasets import make_blobs
import numbers
from sklearn.utils import check_array, check_random_state

from keras.datasets import mnist, fashion_mnist, cifar10


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
        y_w = y[weak_fold]
        Y_w = Y[weak_fold]

        # Select the true labels fold
        X_t = X[true_fold]
        z_t = z[true_fold]
        Z_t = Z[true_fold]
        y_t = y[true_fold]
        Y_t = Y[true_fold]

    # TODO refactor name convention of train and val, for weak and true
    training = (X_w, Z_w, z_w, Y_w, y_w)
    validation = (X_t, Z_t, z_t, Y_t, y_t)
    test = None
    categories = classes
    return training, validation, test, categories


def load_webs(random_state=None, standardize=True, tfidf=False,
              categories=['blog', 'inmo', 'parking', 'b2c', 'no_b2c', 'Other'],
              folder='data/'):
    n_cat = len(categories)

    # Note that the pickle file contains a multi-index dataframe. We take the
    # label (multi-)column only.
    dfY = pd.read_csv(folder + 'Y_newlabels.csv', index_col=0)
    Y_val = dfY[categories].as_matrix()
    y_val = Y_val.argmax(axis=1)

    if np.any(np.sum(Y_val, axis=1) != 1):
        print("Oops!, there are unexpected weak labels in the new dataset."
              "This may cause some errors below")

    X = np.load(folder + 'X_full.npy', encoding='latin1')
    X = X.item()

    # Label dataframe
    dfZ = pd.read_csv(folder + 'Z_full.csv', index_col=0)

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

    # Remove samples where all the weak labels are 0
    index_train_no_0 = (z_train != 0)
    X_train = X_train[index_train_no_0]
    Z_train = Z_train[index_train_no_0]
    z_train = z_train[index_train_no_0]
    index_valid_no_0 = (z_val != 0)
    X_val = X_val[index_valid_no_0]
    Z_val = Z_val[index_valid_no_0]
    z_val = z_val[index_valid_no_0]
    Y_val = Y_val[index_valid_no_0]
    y_val = y_val[index_valid_no_0]

    # Shuffle dataset
    X_train, Z_train, z_train, = shuffle(
        X_train, Z_train, z_train, random_state=random_state)
    X_val, Z_val, z_val, Y_val, y_val = shuffle(
        X_val, Z_val, z_val, Y_val, y_val, random_state=random_state)

    if tfidf:
        tfidf_model = TfidfTransformer()
        X_train = tfidf_model.fit_transform(X_train)
        X_val = tfidf_model.transform(X_val)

    # TODO is it possible to keep the matrices sparse with Keras?
    if sparse.issparse(X_train):
        X_train = X_train.toarray()
        Z_train = Z_train
        z_train = z_train
        X_val = X_val.toarray()

    if standardize:
        scaler_model = StandardScaler()
        X_train = scaler_model.fit_transform(X_train)
        X_val = scaler_model.transform(X_val)

    training = (X_train, Z_train, z_train, None, None)
    validation = (X_val, Z_val, z_val, Y_val, y_val)
    test = None
    return training, validation, test, categories


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

    (X_train, X_val, Z_train, Z_val, z_train, z_val, Y_train, Y_val, y_train,
     y_val) = train_test_split(
         X, Z, z, Y, y, test_size=0.5, random_state=random_state)

    training = (X_train, Z_train, z_train, Y_train, y_train)
    validation = (X_val, Z_val, z_val, Y_val, y_val)
    test = None
    categories = range(0, n_classes)
    return training, validation, test, categories


def load_weak_blobs(method='quasi_IPL', n_samples=2000, n_features=2,
                    centers=6, random_state=None, true_size=0.1, M=None,
                    alpha=0.5, beta=0.3, **kwargs):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=centers, random_state=random_state, **kwargs)

    if M is None:
        if isinstance(centers, int):
            n_classes = centers
        else:
            n_classes = len(centers)
        M = computeM(n_classes, method=method, alpha=alpha, beta=beta,
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


def load_classification(method='quasi_IPL', n_samples=1000, n_features=20,
                        n_classes=6, random_state=None, n_informative=10,
                        n_redundant=2, n_repeated=0, n_clusters_per_class=2,
                        alpha=0.5, beta=0.3, true_size=0.1, M=None):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, random_state=random_state,
                               n_redundant=n_redundant,
                               n_informative=n_informative,
                               n_clusters_per_class=n_clusters_per_class)

    if M is None:
        M = computeM(n_classes, method=method, alpha=alpha, beta=beta,
                     seed=random_state)

    return make_weak_true_partition(M, X, y, true_size=true_size,
                                    random_state=random_state)
    # if n_classes == 2:
    #     Y = np.vstack([1-y, y]).T
    # else:
    #     Y = label_binarize(y, range(n_classes))
    # # FIXME z should not be exactly y
    # Z = Y
    # p2 = np.array([2**n for n in reversed(range(n_classes))])
    # z = Z.dot(p2)

    # X_train, X_val, Z_train, Z_val, z_train, z_val, Y_train, Y_val, y_train, y_val = train_test_split(X, Z, z, Y, y, test_size=0.5, random_state=random_state)

    # training = (X_train, Z_train, z_train)
    # validation = (X_val, Z_val, z_val, Y_val, y_val)
    # test = None
    # categories = range(0, n_classes)
    # return training, validation, test, categories


def load_labelme(random_state=None, prop_valid=0.1, prop_test=0.2,
                 keep_valid_test=False):
    '''
    Loads the LabelMe dataset and reasigns de data in the following manner:
        - Training data divided by two portions -> train and valid
        - Valid and test are join to create test data

    Parameters
    ==========
    random_state: integer
        Random seed to use every time that we shuffle or split the data. If
        None, the samples are not shuffled.

    prop_valid: float (between 0 and 1)
        Proportion of samples from the final training size to use as validation

    prop_test: float (between 0 and 1)
        Proportion of samples from the final training size to use as test

    keep_valid_test: bool
        If true, uses the original validation and test as a test set
        If false, discards the original valid and test sets and creates new
        ones from the training set

    '''
    n_classes = 8
    categories = ['highway', 'insidecity', 'tallbuilding', 'street',
                  'forest', 'coast', 'mountain', 'opencountry']

    # =================================================== #
    # Load train with majority voting
    # =================================================== #
    # Load features (I think these are hidden activations in a VGG16 network)
    X_train = np.load('data/LabelMe/prepared/data_train_vgg16.npy')
    X_train = X_train.reshape(X_train.shape[0], -1)

    # TODO see difference between labels_train_mv and labels_train
    #y_train_mv = np.load('data/LabelMe/prepared/labels_train_mv.npy')
    y_train = np.load('data/LabelMe/prepared/labels_train.npy')
    Y_train = label_binarize(y_train, range(n_classes))
    y_answers = np.load('data/LabelMe/prepared/answers.npy')

    # Convert answers to binary weak labels
    Z_train = np.zeros((X_train.shape[0], n_classes)).astype(int)
    for i, answer in enumerate(y_answers):
        voted = np.unique(answer)
        for v in voted:
            if v != -1:
                Z_train[i, v] = 1

    z_train = weak_to_decimal(Z_train)

    # =================================================== #
    # Create test set
    # =================================================== #
    if keep_valid_test:
        # =================================================== #
        # Load valid and test and join to create test
        # =================================================== #
        X_valid = np.load('data/LabelMe/prepared/data_valid_vgg16.npy')
        X_test = np.load('data/LabelMe/prepared/data_test_vgg16.npy')
        X_valid = X_valid.reshape(X_valid.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_test = np.concatenate((X_valid, X_test))

        y_valid = np.load('data/LabelMe/prepared/labels_valid.npy')
        y_test = np.load('data/LabelMe/prepared/labels_test.npy')
        y_test = np.concatenate((y_valid, y_test))
        Y_test = label_binarize(y_test, range(n_classes))
    else:
        # Separate the augmented samples into different sets
        text_answers = np.array([','.join(y.astype(str)) for y in y_answers])
        unique_answers = np.unique(text_answers)
        y_unique = y_train[[np.where(text_answers == u)[0][0] for u in
                            unique_answers]]

        # =================================================== #
        # Get portion of train for test
        # =================================================== #
        sss = StratifiedShuffleSplit(n_splits=1, random_state=random_state,
                                     train_size=(1. - prop_test),
                                     test_size=prop_test)
        train_indx, test_indx = next(sss.split(unique_answers, y_unique))

        train_indx = np.concatenate(
            [np.where(text_answers == unique_answers[i])[0]
             for i in train_indx])
        test_indx = np.concatenate(
            [np.where(text_answers == unique_answers[i])[0]
             for i in test_indx])

        X_test = X_train[test_indx]
        Y_test = Y_train[test_indx]
        y_test = y_train[test_indx]
        X_train = X_train[train_indx]
        Z_train = Z_train[train_indx]
        z_train = z_train[train_indx]
        Y_train = Y_train[train_indx]
        y_train = y_train[train_indx]
        y_answers = y_answers[train_indx]

    # Separate the augmented samples into different sets
    text_answers = np.array([','.join(y.astype(str)) for y in y_answers])
    unique_answers = np.unique(text_answers)
    y_unique = y_train[[np.where(text_answers == u)[0][0] for u in
                        unique_answers]]

    # =================================================== #
    # Divide training between train and validation
    # =================================================== #
    sss = StratifiedShuffleSplit(n_splits=1, random_state=random_state,
                                 train_size=(1. - prop_valid),
                                 test_size=prop_valid)
    train_indx, val_indx = next(sss.split(unique_answers, y_unique))
    train_indx = np.concatenate(
        [np.where(text_answers == unique_answers[i])[0]
         for i in train_indx])
    val_indx = np.concatenate(
        [np.where(text_answers == unique_answers[i])[0]
         for i in val_indx])

    X_val, Z_val, z_val = X_train[val_indx], Z_train[val_indx], z_train[val_indx]
    Y_val, y_val = Y_train[val_indx], y_train[val_indx]
    X_train, Z_train, z_train = X_train[train_indx], Z_train[train_indx], z_train[train_indx]
    Y_train, y_train = Y_train[train_indx], y_train[train_indx]

    training = (X_train, Z_train, z_train, Y_train, y_train)
    validation = (X_val, Z_val, z_val, Y_val, y_val)
    test = (X_test, Y_test, y_test)
    return training, validation, test, categories


def apply_weak_classifier(X, y, clf=LogisticRegression(), threshold='uniform',
                          true_proportion=0.2, random_state=42):
    '''
    Creates weak and true partition by training a classifier in the weak set,
    and applying a threshold in the predictions of the full set
    '''
    n_c = len(np.unique(y))
    classes = range(0, n_c)
    assert(n_c == np.max(y)+1)
    Y = label_binarize(y, classes)

    # Train a model to create the weak labels
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    # Create partition for weak and true labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=true_proportion,
                                 random_state=random_state)
    for weak_fold, true_fold in sss.split(X, y):
        # Weaken the true labels fold using the mixing matrix M
        X_w = X[weak_fold]
        Y_w = Y[weak_fold]
        y_w = y[weak_fold]

        clf.fit(X_w, y_w)
        p = clf.predict_proba(X)

        # Compute the weak labels in binary form
        if threshold == 'uniform':
            Z = p >= 1.0/Y.shape[1]
        elif isinstance(threshold, float):
            Z = p >= threshold
        else:
            Z = (p.T >= p.max(axis=1)).T

        # Create the weak labels in index form
        z = weak_to_index(Z, method='Mproper')

        z_w = z[weak_fold]
        Z_w = Z[weak_fold]

        # Select the true labels fold
        X_t = X[true_fold]
        Y_t = Y[true_fold]
        y_t = y[true_fold]
        z_t = z[true_fold]
        Z_t = Z[true_fold]

    training = (X_w, Z_w, z_w, Y_w, y_w)
    validation = (X_t, Z_t, z_t, Y_t, y_t)
    test = None
    categories = classes
    return training, validation, test, categories


def load_dataset_apply_model(dataset, **kwargs):
    if dataset == 'iris':
        data = load_iris()
        X = data.data
        y = data.target
    elif dataset == 'digits':
        X, y = load_digits(return_X_y=True)
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
    else:
        raise KeyError(dataset)

    return apply_weak_classifier(X, y, **kwargs)
