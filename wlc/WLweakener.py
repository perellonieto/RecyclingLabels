#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External modules
import numpy as np
# from numpy import binary_repr
import sklearn.datasets as skd           # Needs version 0.14 or higher
from sklearn.preprocessing import label_binarize
# import sklearn.linear_model as sklm
import sys
# import ipdb
from scipy import sparse

import time
import pandas as pd
from scipy.sparse import csr_matrix

from copy import deepcopy


def computeM(c, alpha=0.5, beta=0.5, gamma=0.5, method='supervised', seed=None):
    """
    Generate a mixing matrix M, given the number of classes c.

    Parameters
    ----------
    c      : int. Number of classes
    alpha  : float, optional (default=0.5)
    beta   : float, optional (default=0.5)
    gamma  : float, optional (default=0.5)
    method : string, optional (default='supervised'). Method to compute M.
             Available options are:
                'supervised':   Identity matrix. For a fully labeled case.
                'noisy':        For a noisy label case: the true label is
                                observed with probabiltity 1 - beta, otherwise
                                one noisy label is taken at random.
                'random_noise': All values of the mixing matrix are taken at
                                random from a uniform distribution. The matrix
                                is normalized to be left-stochastic
                'IPL':          Independent partial labels: the observed labels
                                are independent. The true label is observed
                                with probability alfa. Each False label is
                                observed with probability beta.
                'IPL3':         It is a generalized version of IPL, but only
                                for c=3 classes and alpha=1: each false label
                                is observed with a different probability.
                                Parameters alpha, beta and gamma represent the
                                probability of a false label for each column.
                'quasi_IPL':    This is the quasi independent partial label
                                case discussed in the paper.

    Returns
    -------
    M : array-like, shape = (n_classes, n_classes)
    """
    if seed is not None:
        np.random.seed(seed)

    if method == 'supervised':

        M = np.eye(c)

    elif method == 'noisy':

        M = (np.eye(c) * (1 - beta - beta/(c-1)) +
             np.ones((c, c)) * beta/(c-1))

    elif method == 'random_noise':

        M = np.random.rand(c, c)
        M = M / np.sum(M, axis=0, keepdims=True)

        M = (1-beta) * np.eye(c) + beta * M

    elif method == 'random_weak':

        # Number or rows. Equal to 2**c to simulate a scenario where all
        # possible binary label vectors are possible.
        d = 2**c

        # Supervised component: Identity matrix with size d x c.
        Ic = np.zeros((d, c))
        for i in range(c):
            Ic[2**(c-i-1), i] = 1

        # Weak component: Random weak label proabilities
        M = np.random.rand(d, c)
        M = M / np.sum(M, axis=0, keepdims=True)

        # Averaging supervised and weak components
        M = (1-beta) * Ic + beta * M

    elif method == 'IPL':

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(0, d):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = (alpha**(z_bin) * (1-alpha)**(1-z_bin) *
                       (beta**(modz-z_bin) * (1-beta)**(c-modz-1+z_bin)))

        # This is likely not required: columns in M should already sum up to 1
        M = M / np.sum(M, axis=0)

    elif method == 'IPL3':

        b0 = beta[0]
        b1 = beta[1]
        b2 = beta[2]

        M = np.array([
                [0.0,       0.0,       0.0],
                [0,         0,         (1-b2)**2],
                [0,         (1-b1)**2, 0],
                [0.0,       b1*(1-b1), b2*(1-b2)],
                [(1-b0)**2, 0,         0],
                [b0*(1-b0), 0.0,       b2*(1-b2)],
                [b0*(1-b0), b1*(1-1),  0.0],
                [b0**2,     b1**2,     b2**2]])

    elif method == 'quasi_IPL':

        # Convert beta to numpy array
        if isinstance(beta, (list, tuple, np.ndarray)):
            # Make sure beta is a numpy array
            beta = np.array(beta)
        else:
            beta = np.array([beta] * c)

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(1, d-1):

            # Convert the decimal value z to a binary list of length c
            z_bin = [int(b) for b in bin(z)[2:].zfill(c)]
            modz = sum(z_bin)

            M[z, :] = z_bin*(beta**(modz-1) * (1-beta)**(c-modz))

        # Columns in M should sum up to 1
        M = M / np.sum(M, axis=0)

    else:
        raise ValueError("Unknown method to compute M: {}".format(method))

    return M


def generateWeak(y, M, dec_labels=None, seed=None):
    """
    Generate the set of weak labels z from the ground truth labels y, given
    a mixing matrix M and, optionally, a set of possible weak labels, zset.

    Args:
        y       :List of true labels with values from 0 to c-1 where c is the
                 number of classes
        M       :Mixing matrix of shape (d, c) with d >= c.
        dec_labels :A list of indices in {0, 1, ..., 2**c}: dec_labels[i] is an
                 integer whose binary representation encodes the weak labels
                 corresponding to the i-th row in M. The length of dec_labels
                 must be equal to the number of rows in M.

                 If dec_labels is None: the following is assumed:
                   - If M is (2**c, c), dec_labels = [0, 1, ..., 2**c]
                   - If M is (c, c),    dec_labels = [1, 2, 4,..., 2**(c-1)]
                   - Otherwise, a error is raised.

    Returns:
        z   :List of weak labels. Each weak label is an integer whose binary
            representation encodes the observed weak labels.
    """
    if seed is not None:
        np.random.seed(seed)

    z = np.zeros(y.shape, dtype=int)  # Weak labels for all labels y (int)
    d = M.shape[0]               # Number of weak labels
    c = M.shape[1]

    if dec_labels is None:
        if d == c:
            dec_labels = 2**np.arange(c-1, -1, -1)
        else:
            dec_labels = np.arange(d)

    # dec_labels = np.arange(d)    # Possible weak labels (int)
    for index, i in enumerate(y):
        z[index] = np.random.choice(dec_labels, 1, p=M[:, i])

    # if c == d:
    #     z = 2**(c-z-1)

    return z


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


def computeVirtual(z, c, method='IPL', M=None, dec_labels=None):
    """
    Generate the set of virtual labels v for the (decimal) weak labels in z,
    given a weak label model in variable method and, optionally, a mixing
    matrix M, and a list of admissible decimal labels.

    Args:
        z       :List of weak labels or binary matrix (n_samples, n_classes)
                - if List: Each weak label is an integer whose binary
                           representation encondes the observed weak labels
                - if Matrix: Each column represents one class
        c       :Number of classes. All components of z must be smaller than
                 2**c
        method  :Method applied to compute the virtual label vector v.
                 Available methods are:
                 - 'supervised' :Takes virtual label vectors equal to the
                                 binary representations of the weak labels in z

                 - 'IPL'        :Independet Partial Labels. Equivalent to
                                 supervised
                 - 'quasi_IPL'  :Computes virtual labels assuming that the
                                 mixing matrix M was 'quasi_IPL' without
                                 knowing the M
                 - 'Mproper'    :Computes virtual labels for a M-proper loss.
                 - 'MCC'        :Computes virtual labels for a M-CC loss
                                 (Not available yet)
        M       :Mixing matrix. Only for methods 'Mproper' and 'MCC'
        dec_labels :A list of indices in {0, 1, ..., 2**c}: dec_labels[i] is an
                 integer whose binary representation encodes the weak labels
                 corresponding to the i-th row in M. The length of dec_labels
                 must be equal to the number of rows in M.

                 If dec_labels is None: the following is assumed:
                   - If M is (2**c, c), dec_labels = [0, 1, ..., 2**c]
                   - If M is (c, c),    dec_labels = [1, 2, 4,..., 2**(c-1)]
                   - Otherwise, a error is raised.

    Returns:
        v
    """
    v = None
    if len(z.shape) > 1 and z.shape[1] >= 2:
        v = deepcopy(z).astype(float)
        z = weak_to_decimal(z)

    if method in ['supervised', 'IPL']:
        if v is None:
            v = binarizeWeakLabels(z, c).astype(float)
    elif method == 'quasi_IPL':    # quasi-independent labels

        # The virtual labels are computed from the weak label vectors
        if v is None:
            v = binarizeWeakLabels(z, c).astype(float)

        # Each 1 or 0 in the weak label vector must be replaced by a number
        # that depends on the total number of 1's in the vector
        for index in range(len(v)):
            aux = v[index, :]
            weak_sum = np.sum(aux)
            if weak_sum != c:
                weak_zero = float(1-weak_sum)/(c-weak_sum)
                aux[aux == 0] = weak_zero
                v[index, :] = aux
            else:
                # In the quasi_IPL method, it is assumed that nor z=0 nor
                # z=2**C will happen. A zero vector is assigned here, just in
                # case, though the choice is arbitrary.
                # TODO MPN I changed Nans to zeros. Is this important?
                v[index, :] = np.array([None] * c)

    elif method == 'Mproper':
        # Compute array of all possible weak label vectors (in decimal format)
        # in the appropriate order, if not given.
        if dec_labels is None:
            if M.shape[0] == 2**c:
                # All possible weak labels have a row in M
                dec_labels = np.arange(2**c)
            elif M.shape[0] == c:
                # Single-class label vectors are assumed
                dec_labels = 2**np.arange(c-1, -1, -1)
            else:
                dec_labels = np.arange(M.shape[0])
                # 3 raise ValueError("Weak labels for the given M are unknown")

        # Compute inverted index from decimal labels to position in dec_labels
        z2i = dict(zip(dec_labels, range(len(dec_labels))))

        if sparse.issparse(M):
            M = M.toarray()

        # Compute the virtual label matrix
        Y = np.linalg.pinv(M)

        # THIS IS NO LONGER REQUIRD
        # If mixing matrix is square, weak labels need to be transformed from
        # 2**c to c optional values
        # if M.shape[0] == M.shape[1]:
        #     z = c-np.log2(z)-1

        # Compute the virtual label.
        v = np.zeros((len(z), c))
        for i, zi in enumerate(z):
            # The virtual label for the i-th weak label, zi, is the column
            # in Y corresponding to zi (that is taken from the inverted index)
            v[i, :] = Y[:, z2i[zi]].flatten()

    else:
        raise ValueError(
            "Unknown method to create virtual labels: {}".format(method))

    return v


def main():

    # #########################################################################
    # ## MAIN #################################################################
    # #########################################################################

    ############################
    # ## Configurable parameters

    # Parameters for sklearn synthetic data
    ns = 100    # Sample size
    nf = 2      # Data dimension
    c = 3       # Number of classes

    #####################
    # ## A title to start

    print("=======================")
    print("    Weak labels")
    print("=======================")

    ###########################################################################
    # ## PART I: Load data (samples and true labels)                         ##
    ###########################################################################

    X, y = skd.make_classification(
        n_samples=ns, n_features=nf, n_informative=2, n_redundant=0,
        n_repeated=0, n_classes=c, n_clusters_per_class=1, weights=None,
        flip_y=0.0001, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
        shuffle=True, random_state=None)

    M = computeM(c, alpha=0.5, beta=0.5, method='quasi_IPL')
    z = generateWeak(y, M, c)
    v = computeVirtual(z, c, method='quasi_IPL')

    print(M)
    print(M)
    print(M)

    ipdb.set_trace()


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
    if type(dfZ) == pd.DataFrame:
        Z = dfZ[categories].as_matrix()
    else:
        Z = dfZ

    # Initialize (and maybe regularize) the counting matrix
    if reg is None:
        S = csr_matrix((2**n_cat, n_cat))
    elif reg == 'Complete':
        S = csr_matrix(np.ones((2**n_cat, n_cat)))
    elif reg == 'Partial':
        S = csr_matrix((2**n_cat, n_cat))
        weak_list = list(set(Z.dot(p2)))    # Flag vector of existing weak labels
        S[weak_list, :] = 1

    # Convert weak label dataframe into matrix
    if type(dfY) == pd.DataFrame:
        Y = dfY[categories].as_matrix()
    else:
        Y = dfY

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


def newWeakCount(Z, Y, categories, reg=None):
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
        S = csr_matrix(np.ones((2**n_cat, n_cat)))
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


def weak_to_decimal(z):
    """
    >>> import numpy as np
    >>> z = np.array([[ 0.,  0.,  0.,  1.],
    ...               [ 0.,  0.,  1.,  0.],
    ...               [ 1.,  0.,  0.,  0.]])
    >>> weak_to_decimal(z)
    array([1, 2, 8])
    """
    n, n_cat = z.shape
    p2 = np.array([2**n for n in reversed(range(n_cat))])
    return np.array(z.dot(p2), dtype=int)


def estimate_M(Z, Y, categories, reg=None):
    S0 = newWeakCount(Z, Y, categories, reg=reg)
    return S0 / np.sum(S0, axis=0)


def bin_array_to_dec(bitlist):
    """
    >>> bin_array_to_dec([0, 0, 0, 0])
    0
    >>> bin_array_to_dec([0, 0, 0, 1])
    1
    >>> bin_array_to_dec([0, 1, 0, 0])
    4
    >>> bin_array_to_dec([1, 1, 1, 0])
    14
    """
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def weak_to_index(z, method='supervised'):
    """ Index position of weak labels in the corresponding mixing matrix

    It returns the row from the corresponding mixing matrix M where the weak
    label must be. For a supervised method the mixing matrix is a diagonal
    matrix withthe first row belonging to the first class and the last row
    belonging to the last class.

    With an Mproper, IPL, quasiIPL methods the mixing matrix is assumed to be
    2**#classes, where the first row corresponds to a weak labeling with all
    the labels to zero. The second row corresponds to the first class, and the
    last row corresponds to all the classes to one.


    >>> import numpy as np
    >>> z = np.array([[ 0.,  0.,  0.,  1.],
    ...               [ 0.,  0.,  1.,  0.],
    ...               [ 1.,  0.,  0.,  0.]])
    >>> weak_to_index(z, method='supervised')
    array([3, 2, 0])
    >>> weak_to_index(z, method='Mproper')
    array([1, 2, 8])
    >>> z = np.array([[ 0.,  0.,  0.,  0.],
    ...               [ 0.,  1.,  0.,  0.],
    ...               [ 1.,  0.,  1.,  1.]])
    >>> weak_to_index(z, method='Mproper')
    array([ 0,  4, 11])
    """
    c = z.shape[1]
    if method in ['supervised']:
        # FIXME which of both is correct?
        index = np.argmax(z, axis=1)
        #index = c - np.argmax(z, axis=1) - 1
    else:
        #index = np.array(map(bin_array_to_dec, z.astype(int)))
        index = weak_to_decimal(z)
    return index


if __name__ == "__main__":
    import doctest
    doctest.testmod()
