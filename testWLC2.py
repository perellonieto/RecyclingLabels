#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code evaluates logistig regression with weak labels

    Author: JCS, June, 2018
"""

# External modules
import warnings

import numpy as np
import sklearn.datasets as skd
# import sklearn.linear_model as sklm
from sklearn.preprocessing import StandardScaler

# My modules
import wlc.WLclassifier as wlc
import wlc.WLweakener as wlw

from testUtils import plot_data, plot_results, evaluateClassif
import matplotlib.pyplot as plt

import ipdb

warnings.filterwarnings("ignore")
np.random.seed(42)


###############################################################################
# ## MAIN #####################################################################
###############################################################################

############################
# ## Configurable parameters

# Parameters for sklearn synthetic data
nsup = 20          # Number of clean labels
nuns = 20         # Number of weak labels
ntst = 2000         # Number of test samples
nf = 3              # Data dimension
n_classes = 3      # Number of classes
problem = 'blobs'   # 'blobs' | 'gauss_quantiles'

# Common parameters for all algorithms
n_sim = 100         # No. of simulation runs to average
n_jobs = -1        # Number of CPUs to use (-1 means all CPUs)
loss = 'square'    # Loss function: square (brier score) or CE (cross entropy)

# Parameters of the classiffier fit method
# rho = float(1)/5000        # Learning step
ns = nsup + nuns + ntst    # Total number of labels
# n_it = 1000*(nsup + nuns)     # Number of iterations

# Parameters of the weak label model
alpha = 0.6
beta = 0.4
gamma = 0.4
wl_model = 'quasi_IPL'

# Virtual label models
method = 'quasi_IPL'    # 'IPL' | 'quasi_IPL' | 'random_noise' | 'noisy'
method2 = 'Mproper'
# method = 'quasi_IPL_old'

# Constants:
I = np.eye(n_classes)

#####################
# ## A title to start

print("======================================")
print("    Testing Learning from Weak Labels ")
print("======================================")

###############################################################################
# ## PART I: Load data (samples and true labels)                             ##
###############################################################################

# #############
# Input samples


# X, y = skd.make_classification(
#     n_samples=ns, n_features=nf, n_informative=3, n_redundant=0,
#     n_repeated=0, n_classes=n_classes, n_clusters_per_class=2, weights=None,
#     flip_y=0.0001, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
#     shuffle=True, random_state=None)
if problem == 'blobs':
    X, y = skd.make_blobs(n_samples=ns, n_features=nf, centers=n_classes,
                          cluster_std=2.0, center_box=(-10.0, 10.0),
                          shuffle=True, random_state=None)
elif problem == 'gauss_quantiles':
    X, y = skd.make_gaussian_quantiles(n_samples=ns, n_features=nf,
                                       n_classes=n_classes, shuffle=True,
                                       random_state=None)
else:
    raise("Problem type unknown: {}".format(problem))
X = StandardScaler().fit_transform(X)

# ###########
# Weak labels

# Generate composed mixing matrix
M = wlw.computeM(n_classes, alpha=alpha, beta=beta, gamma=gamma,
                 method=wl_model)

# Generate weak labels
z = wlw.generateWeak(y, M)

# Compute virtual labels to explore
v = wlw.computeVirtual(z, n_classes, method=method)
v2 = wlw.computeVirtual(z, n_classes, method=method2, M=M)

# Convert z to a list of binary lists (this is for the OSL alg)
# MPN: I think this needs to be changed to the new binarizeWeakLabels
z_bin = wlw.binarizeWeakLabels(z, n_classes)
# Same with y
y2z = [2**(n_classes-i-1) for i in range(n_classes)]
y_bin = wlw.binarizeWeakLabels(np.array([y2z[yi] for yi in y]), n_classes)

# If dimension is 2, we draw a scatterplot
if nf == 2:
    plot_data(X, y)

######################
# ## Select classifier

# ## Report data used in the simulation
print('----------------')
print('Simulation data:')
print('    Sample size: n = {0}'.format(ns))
print('    Data dimension = {0}'.format(X.shape[1]))

############################################################################
# ## PART II: AL algorithm analysis                                      ###
############################################################################

print('----------------------------')
print('Weak Label Analysis')

wLR = {}
title = {}
v_dict = {}
Pe_tr = {}
Pe_cv = {}
Pe_tst = {}
Pe_tr_mean = {}
Pe_cv_mean = {}
Pe_tst_mean = {}
# params = {'rho': rho, 'n_it': n_it, 'loss': loss}
params = {'loss': loss}
tag_list = []


Xsup = X[:nsup]
ysup = y[:nsup]
Xweak = X[nsup:nsup+nuns]
yweak = y[nsup:nsup+nuns]
zweak = z[nsup:nsup+nuns]
z_bin_weak = z_bin[nsup:nsup+nuns]
Xtst = X[nsup+nuns:]
ytst = y[nsup+nuns:]

# ###################
# Supervised learning
tag = 'Supervised'
title[tag] = 'Supervised learning'
wLR[tag] = wlc.WeakLogisticRegression(
    n_classes, method='OSL', optimizer='BFGS', params=params)
tag_list.append(tag)

# ##################################
# Optimistic Superset Learning (OSL)
tag = 'OSL'
title[tag] = 'Optimistic Superset Loss (OSL)'
wLR[tag] = wlc.WeakLogisticRegression(
    n_classes, method='OSL', optimizer='BFGS')
tag_list.append(tag)

# # ############################################
# # Add hoc M-proper loss with Gradient Descent
tag = 'Mproper'
title[tag] = 'M-proper loss'
wLR[tag] = wlc.WeakLogisticRegression(
    n_classes, method='VLL', optimizer='BFGS', params=params)
tag_list.append(tag)

# ############################################
# Virtual Label Learning with Gradient Descent
tag = 'VLL'
title[tag] = 'Virtual Label Learning (VLL) with regularization'
params = {'alpha': (2.0 + nf)/2, 'loss': loss}   # This alpha is an heuristic
wLR[tag] = wlc.WeakLogisticRegression(
    n_classes, method='VLL', optimizer='BFGS', params=params)
tag_list.append(tag)

# ############################################
# Virtual Label Learning with Gradient Descent
tag = 'VLLc'
title[tag] = 'CC-VLL'
# params = {'rho': rho, 'n_it': n_it, 'loss': loss}
params = {'loss': loss}
wLR[tag] = wlc.WeakLogisticRegression(
    n_classes, method='VLL', optimizer='BFGS', params=params)
v_dict[tag] = z_bin

tag_list.append(tag)
# #################################
# Evaluation and plot of each model

n_cases = 10
n_lab = np.round(np.linspace(float(nsup)/n_cases, nsup, n_cases)).astype(int)

for tag in tag_list:
    Pe_tr[tag] = np.zeros(np.shape(n_lab))
    Pe_cv[tag] = np.zeros(np.shape(n_lab))
    Pe_tst[tag] = np.zeros(np.shape(n_lab))

for i, nsup_i in enumerate(n_lab):

    print('--- Exploring {0} labeled samples out of {1}\r'.format(
          nsup_i, nsup), end="")

    # Combined mixing matrix based on the data prorportions
    ns_i = nsup_i + nuns
    qi = float(nsup_i) / ns_i
    Mi = np.vstack((qi*I, (1-qi)*M))

    # Select supervised dataset
    Xsupi = X[:nsup_i]
    ysupi = y[:nsup_i]
    y_bin_supi = y_bin[:nsup_i]

    Xi = np.vstack((Xsupi, Xweak))
    yi = np.concatenate((ysupi, yweak))
    z_bin_i = np.vstack((y_bin_supi, z_bin_weak))
    zi = wlw.generateWeak(yi, Mi)

    for tag in tag_list:
        # print(tag)

        if tag == 'Supervised':
            Pe_tri, Pe_cvi, Pe_tsti = evaluateClassif(
                wLR[tag], Xsupi, ysupi, ysupi, Xtst, ytst, n_sim=n_sim,
                n_jobs=n_jobs, echo='off')
        else:
            # Compute virtual label matrix
            if tag in ['OSL', 'VLLc']:
                vi = np.vstack((y_bin_supi, z_bin[nsup:nsup + nuns]))
            elif tag == 'Mproper':
                vi = wlw.computeVirtual(zi, n_classes, method=method2, M=Mi)
            elif tag == 'VLL':
                vweak = wlw.computeVirtual(zweak, n_classes, method=method)
                vi = np.vstack((y_bin_supi, vweak))

            Pe_tri, Pe_cvi, Pe_tsti = evaluateClassif(
                wLR[tag], Xi, yi, vi, Xtst, ytst, n_sim=n_sim, n_jobs=n_jobs,
                echo='off')

        Pe_tr[tag][i] = np.mean(Pe_tri)
        Pe_cv[tag][i] = np.mean(Pe_cvi)
        Pe_tst[tag][i] = np.mean(Pe_tsti)

# ############
# Print results.
for tag in tag_list:
    Pe_tr_mean[tag] = np.mean(Pe_tr[tag])
    Pe_cv_mean[tag] = np.mean(Pe_cv[tag])
    Pe_tst_mean[tag] = np.mean(Pe_tst[tag])

    print(title[tag])
    print('* Average train error = {0}'.format(Pe_tr_mean[tag]))
    print('* Average cv error = {0}'.format(Pe_cv_mean[tag]))
    print('* Average test error = {0}'.format(Pe_tst_mean[tag]))

# ############
# Plot results

plt.figure()
for tag in tag_list:
    plt.plot(n_lab, Pe_tr[tag], label=tag)
plt.title('Training error rate')
plt.xlabel('No. of labeled samples')
plt.ylabel('Error rate')
plt.legend()

plt.figure()
for tag in tag_list:
    plt.plot(n_lab, Pe_cv[tag], label=tag)
plt.title('Validatio error rate')
plt.xlabel('No. of labeled samples')
plt.ylabel('Error rate')
plt.legend()

plt.figure()
for tag in tag_list:
    plt.plot(n_lab, Pe_tst[tag], label=tag)
plt.title('Test error rate')
plt.xlabel('No. of labeled samples')
plt.ylabel('Error rate')
plt.legend()

print('================')
print('Fin de ejecucion')

plt.ion()
plt.show()
ipdb.set_trace()
