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


def computeM(c, alpha=0.5, beta=0.5, gamma=0.5, method='supervised'):
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

    if method == 'supervised':

        M = np.eye(c)

    elif method == 'noisy':

        M = (np.eye(c) * (alpha - (1-alpha)/(c-1))
             + np.ones((c, c)) * (1-alpha)/(c-1))

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
        for i in xrange(c):
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

        M = np.array([
                [0.0,             0.0,           0.0],
                [0,               0,             (1-gamma)**2],
                [0,               (1-beta)**2,   0],
                [0.0,             beta*(1-beta), gamma*(1-gamma)],
                [(1-alpha)**2,    0,             0],
                [alpha*(1-alpha), 0.0,           gamma*(1-gamma)],
                [alpha*(1-alpha), beta*(1-beta), 0.0],
                [alpha**2,        beta**2,       gamma**2]])

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


def generateWeak(y, M, dec_labels=None):
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

    z = np.zeros(y.shape, dtype=int)  # Weak labels for all labels y (int)
    d = M.shape[0]               # Number of weak labels
    c = M.shape[1]

    if dec_labels is None:
        if d == 2**c:
            dec_labels = np.arange(2**c)
        elif d == c:
            dec_labels = 2**np.arange(c-1, -1, -1)
        else:
            raise ValueError(
                "A dec_labels parameter is required for the given M")

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
        z       :List of weak labels. Each weak label is an integer whose
                 binary representation encondes the observed weak labels
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

    if method in ['supervised', 'IPL']:
        v = binarizeWeakLabels(z, c).astype(float)
    elif method == 'quasi_IPL':    # quasi-independent labels

        # The virtual labels are computed from the weak label vectors
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
                raise ValueError("Weak labels for the given M are unknown")

        # Compute inverted index from decimal labels to position in dec_labels
        z2i = dict(zip(dec_labels, range(len(dec_labels))))

        # Compute the virtual label matrix
        Y = np.linalg.pinv(M)

        # THIS IS NO LONGER REQUIRD
        # If mixing matrix is square, weak labels need to be transformed from
        # 2**c to c optional values
        # if M.shape[0] == M.shape[1]:
        #     z = c-np.log2(z)-1

        # Compute the virtual label.
        v = np.zeros((z.size, c))
        for i, zi in enumerate(z):
            # The virtual label for the i-th weak label, zi, is the column
            # in Y corresponding to zi (that is taken from the inverted index)
            v[i, :] = Y[:, z2i[zi]]
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

    print "======================="
    print "    Weak labels"
    print "======================="

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

    print M
    print z
    print v

    ipdb.set_trace()


if __name__ == "__main__":

    main()

