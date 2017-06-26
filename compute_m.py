import numpy as np
import time
from sklearn.preprocessing import label_binarize
import pandas as pd
from scipy.sparse import csr_matrix


# TODO implement for numpy arrays
def weakCount(dfZ, dfY, categories, reg=None):
    """ Compute the matrix of weak label counts in df_Z for each category in
        df_Y

    Parameters
    ----------
    categories: A list of category names.
    dfZ:        A pandas dataframe indexed by a sample identifier, and
                containing at least one column per category. The weak label
                vector for each sample is composed by the binary values in
                these colummns.
    dfY:        A pandas dataframe indexed by a sample identifier, and
                containing at least one column per category. The weak
                label vector for each sample is composed by the binary
                values in these colummns.
    reg:        Type of regularization:
                    - None:       The weak count is not regularized
                    - 'Partial':  Only rows corresponding to weak
                                  labels existing in dfZ are
                                  regularized (by adding 1)
                    - 'Complete': All valuves are regularized (by adding 1)

    Returns
    -------
    S: A count matrix. S(w, c) contains the number of times that a sample
       appears in df_Z with the weak label vector identified by w and also in
       df_Y with the weak label vector identified by c.  The identifier of a
       label vector is the decimal number corresponding to the binary number
       resulting from the concatenation of its components.  If reg=='Complete',
       the output matrix is dense. Otherwise, it is a sparse_csr matrix.
    """

    # Number of categories
    n_cat = len(categories)

    # These vectors are useful to convert binary vectors into integers.
    # To convert arbitrary binary vectors to a decimal
    p2 = np.array([2**n for n in reversed(range(n_cat))])
    # To convert single-1-vectors to position integers.
    ind = range(n_cat)

    # Convert weak label dataframe into matrix
    Z = dfZ[categories].as_matrix()

    # Initialize (and maybe regularize) the counting matrix
    if reg is None:
        S = csr_matrix((2**n_cat, n_cat))
    elif reg == 'Complete':
        S = np.ones((2**n_cat, n_cat))
    elif reg == 'Partial':
        S = csr_matrix((2**n_cat, n_cat))
        weak_list = list(set(Z.dot(p2)))    # Flag vector of existing weak labels
        S[weak_list, :] = 1

    # Convert weak label dataframe into matrix
    Y = dfY[categories].as_matrix()

    # Start the weak label count
    for idx in dfY.index:

        # True label
        y = dfY.loc[idx].as_matrix()
        c = y.dot(ind)

        # Weak label
        if idx in dfZ.index:
            z = dfZ.loc[idx, categories].as_matrix()
            w = int(z.dot(p2))

            S[w, c] += 1

    return S


def new_weak_count(Z, Y, categories, reg=None):
    # Number of categories
    n_cat = len(categories)

    # These vectors are useful to convert binary vectors into integers.
    # To convert arbitrary binary vectors to a decimal
    p2 = np.array([2**n for n in reversed(range(n_cat))])
    # To convert single-1-vectors to position integers.
    ind = range(n_cat)

    if type(Z) == pd.DataFrame:
        # Convert weak label dataframe into matrix
        Z = Z.as_matrix()

    # Initialize (and maybe regularize) the counting matrix
    if reg is None:
        S = csr_matrix((2**n_cat, n_cat))
    elif reg == 'Complete':
        S = np.ones((2**n_cat, n_cat))
    elif reg == 'Partial':
        S = csr_matrix((2**n_cat, n_cat))
        # Flag vector of existing weak labels
        weak_list = np.unique(np.array(Z.dot(p2)))
        S[weak_list, :] = 1

    if type(Y) == pd.DataFrame:
        # Convert weak label dataframe into matrix
        Y = Y[categories].as_matrix()

    # Start the weak label count
    y_class = np.argmax(Y, axis=1)
    for i, (y, c, z) in enumerate(zip(Y, y_class, Z)):
        # Weak label
        w = int(z.dot(p2))

        S[w, c] += 1

    return S

def compute_M(dfZ, dfY, categories, reg=None):
    S0 = weakCount(dfZ, dfY, categories, reg=reg)
    return S0 / np.sum(S0, axis=0)


def compare_weakCount():
    n_classes = 7
    categories = range(n_classes)
    np.random.seed(0)
    z = np.random.randint(2**n_classes, size=500)
    Z = np.matrix([list(np.binary_repr(x, n_classes)) for x in z], dtype=int)
    y = Z.argmax(axis=1)
    Y = label_binarize(y, categories)
    dfZ = pd.DataFrame(Z)
    dfY = pd.DataFrame(Y)
    for reg in [None, 'Partial', 'Complete']:
        start = time.time()
        wc = weakCount(dfZ, dfY, categories, reg=reg)
        end = time.time()
        print('weakCount time = %s seconds' % (end - start))
        start = time.time()
        wc2 = new_weak_count(Z, Y, categories, reg=reg)
        end = time.time()
        print('new_weak_count time = %s seconds' % (end - start))
        print np.min(wc==wc2)


def test_compute_M():
    n_classes = 4
    categories = range(n_classes)
    np.random.seed(0)
    z = np.random.randint(n_classes**2, size=50)
    Z = np.matrix([list(np.binary_repr(x, n_classes)) for x in z], dtype=int)
    y = Z.argmax(axis=1)
    Y = label_binarize(y, categories)
    dfZ = pd.DataFrame(Z)
    dfY = pd.DataFrame(Y)
    wc = compute_M(dfZ, dfY, categories, reg=None)
    wc = compute_M(Z, dfY, categories, reg=None)
    print wc


if __name__ == '__main__':
    compare_weakCount()
    #test_compute_M()
