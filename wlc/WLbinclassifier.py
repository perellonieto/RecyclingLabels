#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines classifier objects that work with weak labels

    Author: JCS, May. 2016
"""

import numpy as np
# import ipdb


class WeakLogisticRegression(object):

    def __init__(self, n_classes=2, rho=0.1, n_it=100, sound='off'):

        """
        Only a name is needed when the object is created
        """

        self.sound = sound
        self.rho = rho
        self.n_it = n_it
        self.n_classes = n_classes
        self.classes_ = range(n_classes)

    def logistic(self, x):
        """
        Computes the logistic function
        """

        p = 1.0 / (1 + np.exp(-x))
        return p

    def softmax(self, x):
        """
        Computes the softmax transformation

        Args:
            :x  : NxC - matrix of N samples with dimension C

        Returns:
            :p  : NxC - matrix of N probability vectors with dimension C


        """

        p = np.exp(x)
        p = p / np.sum(p, axis=2)

        return p

    def fit(self, Z_tr, Y_tr):
        """
        Fits a logistic regression model to instances in Z_tr given
        the labels in Y_tr

        Args:
            :Z_tr :Input data, numpy array of shape[n_samples, n_features]
            :Y_tr :Target vector relative to Z_tr, with shape [n_samples]
        """

        # Data dimension
        n_dim = Z_tr.shape[1]

        # Add extra dimension if required
        if len(Y_tr.shape) == 1:
            Y_tr = Y_tr[:, None]

        # Initialize variables
        nll_tr = np.zeros(self.n_it)
        # pe_tr = np.zeros(self.n_it)
        w = np.random.randn(n_dim, 1)

        # Running the gradient descent algorithm
        for n in range(self.n_it):

            # Compute posterior probabilities for weight w
            p1_tr = self.logistic(np.dot(Z_tr, w))
            p0_tr = self.logistic(-np.dot(Z_tr, w))

            # Compute negative log-likelihood
            nll_tr[n] = (- np.dot(Y_tr.T, np.log(p1_tr)) -
                         np.dot((1-Y_tr).T, np.log(p0_tr)))

            # Update weights
            w += self.rho*np.dot(Z_tr.T, Y_tr - p1_tr)
        self.w = w

        return self    # w, nll_tr

    def predict(self, Z):

        # Compute posterior probability of class 1 for weights w.
        p = self.logistic(np.dot(Z, self.w))

        # Class
        D = np.array([int(round(pn)) for pn in p])

        return D  # p, D

    def predict_proba(self, Z):

        # Compute posterior probability of class 1 for weights w.
        p = (np.c_[self.logistic(np.dot(-Z, self.w)),
             self.logistic(np.dot(Z, self.w))])
        return p

    def get_params(self, deep=True):

        # suppose this estimator has parameters "alpha" and "recursive"
        return {"sound": self.sound, "rho": self.rho, "n_it": self.n_it}
