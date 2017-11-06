#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Evaluates a specific tourney-based and pool-based active learning (AL)
    algorithm.

    Using sklearn, a dataset is created and a classifier is trained using a
    pool-based active learning algorithm.

    Use: python test_al.py

    Check configurable parameters.

    Author: JCS, Mar. 2016
"""

# External modules
import numpy as np
import sklearn.datasets as skd           # Needs version 0.14 or higher
import sklearn.linear_model as sklm
import ipdb
import matplotlib.pyplot as plt

# My modules
import activelearner as al


def generate_samples(ns, nf):
    """
    This is just a home-made sample generator
    """

    y = np.zeros(ns)
    X = np.zeros((ns, nf))

    for i in range(ns):
        y[i] = 2*np.random.randint(0, 2) - 1

        if y[i] == -1:
            X[i] = np.random.uniform(0, 3, nf)
        else:
            X[i] = np.random.uniform(2, 5, nf)

    return X, y


def compute_sample_eff(Pe_ref, Pe_test, nq):
    """
    Compute the sample efficiency of error rate array Pe_test with respect
    to the reference error rate Pe_ref.
    """

    m = np.zeros(len(Pe_ref))

    for k, pk in enumerate(Pe_ref):

        # Find the pool where AL got an error rate of at most pk
        dif = pk - Pe_test
        i1 = np.argmax(np.array(dif) >= 0)

        if i1 == 0 and dif[0] < 0:
            i1 = len(nq)-1
            i2 = len(nq)-1
        elif dif[i1] == 0 or i1 == 0:
            i2 = i1
        else:
            i2 = i1 - 1

        # Transform pool values into their corresponding number of samples
        m1 = nq[i1]
        m2 = nq[i2]

        q1 = Pe_test[i1]
        q2 = Pe_test[i2]

        # Interpolate m at pk, between q1 and q2
        if q2 != q1:
            m[k] = m1 + (pk-q1)/(q2-q1)*(m2-m1)
        else:
            m[k] = m1

    return m


###############################################################################
# ## MAIN #####################################################################
###############################################################################

############################
# ## Configurable parameters

# Parameters for sklearn synthetic data
ns = 1000    # Sample size
nf = 2      # Data dimension

# Common parameters for all AL algorithms
threshold = 0
pool_size = 20  # No. labels requested at each pool of Active Learner
n_sim = 50    # No. of simulation runs to average

# Type of AL algorithg
type_AL = 'tourney'    # AL algorithm
ts = 40           # Tourney size

#####################
# ## A title to start

print "======================="
print "    Active Learning"
print "======================="

###############################################################################
# ## PART I: Load data (samples and true labels)                             ##
###############################################################################

X, y = skd.make_classification(
    n_samples=ns, n_features=nf, n_informative=2, n_redundant=0,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
    flip_y=0.0001, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
    shuffle=True, random_state=None)

# This is an alternative to artificial set generation.
# from sklearn.datasets.samples_generator import make_blobs
# X, y = make_blobs(n_samples=ns, centers=2, random_state=0, cluster_std=0.60)
y = 2*y - 1   # Convert labels to +-1
# X, y = generate_samples(ns, nf)

# If dimension is 2, we draw a scatterplot
if nf == 2:
    # Scatterplot.
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='copper')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.axis('equal')
    plt.draw()

######################
# ## Select classifier

# Create classifier object
myClassifier = sklm.LogisticRegression(
    penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
    intercept_scaling=1, class_weight=None, random_state=None,
    solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,
    warm_start=False, n_jobs=1)

# ## Report data used in the simulation
print '-----------------------'
print 'Datos de la simulación:'
print '    Tamaño muestral: n = ' + str(ns) + ' muestras, etiquetadas en ',
print str(ns/pool_size) + ' lotes de ' + str(pool_size) + ' muestras cada uno.'
print '    Dimensión de los datos = ' + str(X.shape[1])

############################################################################
# ## PART II: AL algorithm analysis                                      ###
############################################################################

print '----------------------------'
print 'AL analysis'

print 'Evaluando muestreo aleatorio...'
ALrandom = al.ActiveLearner('random')
Pe_random, PeRaw_random, PeW_random = ALrandom.evaluate(
    myClassifier, X, y, pool_size, n_sim)

print 'Evaluando Active Learning...'
ALtourney = al.ActiveLearner(type_AL, p_relabel=0, alth=threshold, p_al=1,
                             tourneysize=ts)
Pe_AL, PeRaw_AL, PeW_AL = ALtourney.evaluate(
    myClassifier, X, y, pool_size, n_sim)

#################
# ## Plot results

# Color codes
color1 = [0.0/255.0, 0.0/255.0, 0.0/255.0]
color2 = [0.0/255.0, 152.0/255.0, 195.0/255.0]
color3 = [177.0/255.0, 209.0/255.0, 55.0/255.0]
color4 = [103.0/255.0, 184.0/255.0, 69.0/255.0]
color5 = [8.0/255.0, 128.0/255.0, 127.0/255.0]
color6 = [46.0/255.0, 50.0/255.0, 110.0/255.0]
color7 = [134.0/255.0, 37.0/255.0, 98.0/255.0]
color8 = [200.0/255.0, 16.0/255.0, 58.0/255.0]
color9 = [194.0/255.0, 53.0/255.0, 114.0/255.0]
color10 = [85.0/255.0, 53.0/255.0, 123.0/255.0]
color11 = [15.0/255.0, 100.0/255.0, 170.0/255.0]
color12 = [68.0/255.0, 192.0/255.0, 193.0/255.0]
color13 = [27.0/255.0, 140.0/255.0, 76.0/255.0]
color14 = [224.0/255.0, 208.0/255.0, 63.0/255.0]
color15 = [226.0/255.0, 158.0/255.0, 47.0/255.0]
color16 = [232.0/255.0, 68.0/255.0, 37.0/255.0]

font = {'family': 'Verdana',
        'weight': 'regular',
        'size': 10}

# matplotlib.rc('font', **font)

# Plot error rates vs labeled samples
print 'Tamaño muestral: n = ' + str(ns) + ' muestras'
print str(ns) + ' muestras, etiquetadas en ' + str(ns/pool_size) \
    + ' lotes de ' + str(pool_size) + ' muestras cada uno.'

# ## Vector of succesive sample sizes
nq = range(pool_size, ns+pool_size, pool_size)
nq[-1] = min(nq[-1], ns)
n_pools = len(nq)

print 'Dimension de la matriz de features = ' + str(X.shape)
print 'Numero de etiquetas = ' + str(ns)
fig = plt.figure()
h1, = plt.plot(nq, Pe_random, '--', marker='o', color=color1)
h2, = plt.plot(nq, Pe_AL, '-', marker='.', color=color2)
plt.legend([h1, h2], ["Random Sampling", "Active Learning"])
fig.suptitle(u'Testing algorithms')
plt.xlabel(u'Labeled dataset size')
plt.ylabel('Error rate (computed over the whole dataset)')
plt.show(block=False)

fig = plt.figure()
h1, = plt.plot(nq, Pe_AL, '-', marker='.', color=color2)
h2, = plt.plot(nq, PeRaw_AL, '-', marker='.', color=color3)
h3, = plt.plot(nq, PeW_AL, '-', marker='.', color=color4)

# Text in the figures
plt.legend([h1, h2, h3],
           ["True error rates", "Raw sampling", "Importance sampling"])
fig.suptitle(u'Evolución del nº de errores con el nº de datos etiquetados')
plt.xlabel(u'Labeled dataset size')
plt.ylabel('Error rate')
plt.draw()
# plt.show(block=False)


######################
# ## Sample efficiency
m = np.zeros(n_pools)
Pe_AL_opt = np.minimum.accumulate(Pe_AL)
Pe_AL_pes = np.maximum.accumulate(Pe_AL[::-1])[::-1]

m_opt = compute_sample_eff(Pe_random, Pe_AL_opt, nq)
m_pes = compute_sample_eff(Pe_random, Pe_AL_pes, nq)
m_opt_ext = np.append(0, m_opt)
m_pes_ext = np.append(0, m_pes)
nq_ext = np.append(0, nq)

fig = plt.figure()
plt.plot(nq_ext, m_pes_ext, '-', marker='.', color=color3)
plt.fill_between(nq_ext, 0, m_pes_ext, color=color3)
plt.plot(nq_ext, m_opt_ext, '-', marker='.', color=color2)
plt.fill_between(nq_ext, 0, m_opt_ext-1, color=color2)
plt.plot(nq_ext, nq_ext, '-', color=color1)
plt.axis([0, max(nq_ext), 0, max(nq_ext)])
fig.suptitle(u'Eficiencia muestral del Active Learning')
plt.xlabel(u'Demanda de etiquetas sin AL')
plt.ylabel(u'Demanda de etiquetas con AL')

plt.show(block=False)

ipdb.set_trace()

print '================'
print 'Fin de ejecucion'
