#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code contains a method to evaluate the performance
    o a weak learning classifier.

    Author: JCS, June, 2016
"""

# External modules
import numpy as np
import sklearn.model_selection as skms
import time
# import ipdb


def evaluateClassif(classif, X, y, v=None, n_sim=1):

    # Default v
    if v is None:
        v = y

    # Sample size and data dimension
    ns, nf = X.shape

    # ## Initialize aggregate results
    Pe_tr = [0] * n_sim
    Pe_cv = [0] * n_sim

    print '    Averaging {0} simulations. Estimated time...'.format(n_sim),

    # ## Loop over simulation runs
    for i in xrange(n_sim):

        if i == 0:
            start = time.clock()

        # ########################
        # Ground truth evaluation:
        classif.fit(X, v)
        f = classif.predict_proba(X)

        # Then, we evaluate this classifier with all labels
        # Note that training and test samples are being used in this error rate
        d = np.argmax(f, axis=1)
        Pe_tr[i] = float(np.count_nonzero(y != d)) / ns

        # ##############
        # Self evaluation.
        # First, we compute leave-one-out predictions
        n_folds = min(10, ns)
        preds = skms.cross_val_predict(classif, X, v, cv=n_folds, verbose=0)

        # Estimate error rates:
        Pe_cv[i] = float(np.count_nonzero(y != preds)) / ns

        if i == 0:
            print '{0} segundos'.format((time.clock() - start) * n_sim)

    return Pe_tr, Pe_cv
