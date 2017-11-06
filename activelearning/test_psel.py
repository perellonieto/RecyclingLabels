#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This script is aimed at evaluating the behavior of tourney_prob, which
    computes the selection probabilities in a set of size N.

    The selection probability p for tuple (k, N, m) is the probability  of
    selecting the k-th best sample in a ranked set of N samples when taking the
    best sample in a random subset of m elements.

    - Tests if the function is correct (by testing if probabilities agree with
      frequencies of occurrence)
    - Evaluates the sample average estimates of E{1/p} and some related
      measures. This is an indicator of the behavior that can be expected from
      average statistics computed from active learning samplers based on
      tournaments (which is the the algorithm used by the WebLabeler
      application)

    Use: python test_psel.py

    Check configurable parameters.

    Author: JCS, Mar. 2016
"""

# External modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import ipdb


def tourney_prob(k, N, m):

    """
    Compute the probability of the following event in the following
    experiment:

    - Experiment: given a set S_N of N distinct real numbers, take subset
      S_m of m < N values at random without replacement.
    - Event: The highest value in S_m is the k-th highest value in S_N,
      given that the k-th higest value is in S_m

    Args:
        :k: Rank to be evaluated
        :N: Size of the original set
        :m: Size of the subset

    Returns:
        :p: Probability of k-th highest value over N being the highest over
            m
    """

    if N < m:
        print "The second argument cannot be smaller than the third one."
        sys.exit()

    if m < 1 or k <= 0:
        return 0.0
    elif m == 1:
        return 1.0 / N
    else:
        return float(N - k) * m / (N * (m - 1)) * tourney_prob(k, N - 1, m - 1)

# Configurable parameters
ts = 10             # Tourney size
n_tot = 400         # Dataset size
n_trials = 40000    # Number of draws for the first test (it can be larger than
# n_tot because we simulate smapling with replacement)

# Configurable parameters for the second test
n_cut = 100      # Size of the selection sequence
n_sim = 600      # Number of simulations
drawplots = True  # True for graphical results (n_sim should not be too large)

# ####################################################
# Testing tourney probabilities and importance weights

print "*****************"
print "* FIRST TEST ***"

print "This is a test to verify that the probabilities computed by ",
print "tourney_prob agree with the frequencies observed in a simulation."

# Testing p_sel
n = n_tot + ts - 1

whist = np.zeros(n)
p_est = np.zeros(n_trials)
winners2 = np.zeros(n_trials)

for k in range(n_trials):

    # Tourney simulation
    players = range(n)
    players = np.random.permutation(players)
    players = players[0:ts]
    winner = min(players)
    whist[winner] += 1
    p_est[k] = tourney_prob(winner + 1, n, ts)
    winners2[k] = winner

p_sel2 = [tourney_prob(k + 1, n, ts) for k in range(n)]
print p_sel2
# whist = whist / sum(whist)
# plt.figure()
# plt.plot(whist)
# plt.plot(p_sel2)

print "Sample size = ", n_trials
print "This is like a repetition of the first test, but using a larger ",
print "number of samples"
print "Minimum prob = ", min([p for p in p_sel2 if p > 0])
p_sel2 = [p for p in p_sel2 if p > 0]     # Remove zero probability values
n = len(p_sel2)
print "Testing if the probabilities computed by tourney_prob are correct"
print "Here we simply compute if E{p} is equal to the sample average of p"
p_mean = sum(p * p for p in p_sel2)
p_std = np.sqrt(sum((p - p_mean)**2 * p for p in p_sel2))
print "Expected p is ", p_mean
print "Standard deviation of p is ", p_std
print "Average p is ", np.mean(p_est)


print "Now we test if E{1/(np)} is close to the sample avergage of 1/p."
print "E{1/p} should be exactly one."
ip_mean = sum(1.0/n for p in p_sel2)
ip_std = np.sqrt(sum(((1.0 / (n*p) - ip_mean)**2)*p for p in p_sel2))
print "Expected 1/(np) is ", ip_mean
print "Std of 1/(np) is ", ip_std
print "Average 1/(np) is ", np.mean(1.0 / (n*p_est))
print "Very likely, you will notice that the std is huge, despite the average",
print " 1/(np) is not so large. This is because, with very low probability,",
print "1/(np) can be very large."


print "*********************************************"
print "* SECOND TEST: tourney_prob PROBABILITIES ***"

print "The second test was aimed at computing the estimates basde on joint ",
print "probabilities. It was used to verify that importance weights based on ",
print "joint probabilities are very bad (huge variance)."
p_sel = np.zeros((n_sim, n_cut))
p_all = np.zeros((n_sim, n_cut))
np_all = np.zeros((n_sim, n_cut))
w_all = np.zeros((n_sim, n_cut))
winners = np.zeros((n_sim, n_cut))

# First we enlarge (virtually) the number of players in the tournament, to
# give a nonzero selection probability to all samples in the dataset.
# (Otherwise, e.g. a tourney of size 5 would never select the 4 worst samples
# in the original set of players)
n_max = n_tot + ts - 1

# Simulation loop
for i in range(n_sim):

    # ### Tourney simulation
    nk = n_max

    for k in range(n_cut):

        # Tourney simulation
        players = np.random.permutation(range(nk))[0:ts]
        winner = min(players)
        winners[i, k] = winner

        # Selection probability for the winner
        p_sel[i, k] = tourney_prob(winner + 1, nk, ts)

        if k == 0:
            p_all[i, k] = p_sel[i, k]
            np_all[i, k] = (nk - ts + 1) * p_sel[i, k]
        else:
            p_all[i, k] = p_sel[i, k] * p_all[i, k - 1]
            np_all[i, k] = (nk - ts + 1) * p_sel[i, k] * np_all[i, k - 1]

        nk -= 1

mean_np = np.mean(np_all, axis=0)
w_all = 1.0 / np_all
mean_w = np.mean(w_all, axis=0)

# Compute true selection probability distribution for the 1st round
p_sel1 = [tourney_prob(k + 1, n_max, ts) for k in range(n_tot)]

z = xrange(n_tot)

if drawplots:
    # plt.figure()
    # plt.plot(p_sel.T)
    # plt.show(block=False)
    # plt.title("Selection probabilities")

    # plt.figure()
    # plt.semilogy(p_all.T)
    # plt.show(block=False)

    # plt.title("Joint probabillities")

    # plt.figure()
    # plt.semilogy(np_all.T)
    # plt.show(block=False)
    # plt.title("Scaled joint probabilities")

    plt.figure()
    plt.plot(mean_np)
    plt.show(block=False)
    plt.title("Mean Scaled joint probabilities")

    # plt.figure()
    # plt.plot(w_all.T)
    # plt.show(block=False)
    # plt.title("Scaled weights")

    plt.figure()
    plt.plot(mean_w)
    plt.show(block=False)
    plt.title("Mean Scaled weights")

print "I am stopping just to keep figures open."
ipdb.set_trace()

