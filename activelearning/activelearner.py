# -*- coding: utf-8 -*-
"""
Defines the class ActiveLearner, which can be used to implement different
active learning algorithms

Created on Wed Jul  2 18:28:28 2014

Last update: Jan 13 2014.

@author: jcid
"""

import math
import numpy as np
import sys
import time
import ipdb
import sklearn.cross_validation as skcv


class ActiveLearner(object):
    """
    ActiveLearner implements different algorithms for active learning (AL).
    Assumes binary classification and classes +1 and -1.
    Active Learning algorithms assume a labeled dataset.
        - Labels -1 and 1 indicates a sample already labeled.
        - Label 0 indicates a candidate sample for class-labeling
        - In other case, the sample is non-eligible for class-labeling.

    An AL object is created using ActiveLearner(name).

    The name contains the type of AL algorithm to be used, among the following:

        'random'  :Samples are selected at random.
        'greedy'  :Take samples in increasing entropy order
        'tourney' :Take groups of samples at random an select the one with the
                   lowest score

    The following parameters can be specified after creating any object from
    this class (the default value is also shown):

        self.alth=0 :AL threshold. This is used by the AL algorithm to compute
                     the score of each sample.
                     A standard choice is to set it equal to the decision
                     threshold of the classifier model, but there may be good
                     reasons for other choices. For instance, is classes are
                     umbalanced, we can take self.alth = 1 so as to promote
                     sampling from the positive class
        self.tourneysize=40 :Size of each tourney (only for 'tourney')
        self.p_al=1  :Probability of AL sampling. Samples are selected using AL
                      with probability p, and using random sampling otherwise.
        self.p_relabel :Probability of selecting a sample already labelled
    """

    # __name = ''
    # __dictUsrs

    def __init__(self, name, p_relabel=0, alth=0.5, p_al=1, tourneysize=40,
                 weights=None, n_classes=2, sound='off'):
        """
        Only a name is needed when the object is created
        """
        supportedALs = {'random', 'greedy', 'tourney'}

        if name in supportedALs:
            self.name = name
        else:
            print 'Active Learning Mode not supported. \
                   Random sampling used instead'
            self.name = 'random'

        self.sound = sound
        self.n_classes = n_classes

        # Set default values
        self.alth = alth
        self.p_relabel = p_relabel

        # Default probability of active learning
        if self.name == 'random':
            # Random sampling is equivalent to AL with zero probability of AL
            # Note that in case of random sampling, input p_al is ignored
            self.p_al = 0
        else:
            self.p_al = p_al

        # No. of samples entering into each tourney
        if self.name == 'tourney':
            self.tourneysize = tourneysize

        # Initialize vector of weights for aggregated measures.
        self.weights = None

    def tourney_prob(self, k, N, m):

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

        Note for developer: this function could be out of the class
        """

        if N < m:
            print "The second argument cannot be smaller than the third one."
            sys.exit()

        if m < 1:
            return 0.0
        elif m == 1:
            return 1.0 / N
        else:
            return float(N - k) / N * self.tourney_prob(k, N - 1, m - 1)

    def get_queries(self, num_queries, labels, s, w=None):
        """
        Select new label queries applying and active learning algorithm

        It applies the active learning algorithm specified in self.name

        Args:
            num_queries: number of queries to be returned.
            labels : vector of current labels
                     = -1 or 1 if the label is known
                     = 0 if the label is unknown, and must be labeled
                     Any other value is processed as if 0
            s      : soft decision of the classifier
            w      : array of weights for each score. If None, it is
                      initialized to zeros.

        Returns:
            query_pool :Vector of indices to the selected samples
            rs0_al1:    Vector indicating which samples have been selected
                        by random sampling (0) or by active learning (1)
            unl0_lab1:  Vector indicating (with 1) which samples already
                        had a label (0 if not)
            w_new      :Array of weights updated by the new pool of queries.
                        Weight update is based on selection probabilities.
                        Selection probabilities are useful tocompute unbiased
                        average measures from the sample set selected by the
                        AL.
        """

        # Dataset size
        ns = len(s)

        # The number of selected queries cannot exceed the dataset size.
        num_queries = min(num_queries, ns)

        # Initizialize weights, if necessary.
        # This function assumes that an array containing one weight for each
        # score is available. If not, it is created as an array of zeros.
        if w is None:
            w = np.zeros(ns)

        # Create a copy of w in w_new, to be modified later.
        # This is likely an old and non-pythonic programming style...
        w_new = np.copy(w)

        #################
        # ## Scoring data

        # Compute unsupervised score
        # Samples close to the threshold get small scores.
        c_unl = abs(np.array(s)-self.alth)
        c_unl = c_unl/max(c_unl + sys.float_info.min)  # (add eps to avoid 0/0)

        # Compute label-based score
        # Samples with highest discrepancy with the true label get small scores
        c_lab = np.array([0.5] * ns)
        ilabeled = [j for j in xrange(len(labels)) if labels[j] is not None]
        c_lab[ilabeled] = 1 - abs(np.array(s)[ilabeled] -
                                  np.array(labels)[ilabeled]) / 2

        ############################
        # ## Selection probabilities

        # Compute selection probabilities assuming a tourney over all samples
        # in the input dataset. That is, they are computed assuming that all
        # samples in the dataset are available and unlabeled.
        if self.name == "random":
            rank2p_sel = [1.0 / ns for k in range(1, ns + 1)]
        elif self.name == "tourney" or self.name == "greedy":
            ts_all = min(ns, self.tourneysize)
            rank2p_sel = [((1.0 - self.p_al) / ns + self.p_al * ts_all *
                           self.tourney_prob(k, ns + ts_all - 1, ts_all))
                          for k in range(1, ns + 1)]

        #####################################
        # ## Split labeled and unlabeled data

        # Labeled data flag
        islabeled = [l is not None for l in labels]

        # Rank queries by their unsupervised score
        rank2query = np.argsort(c_unl)       # Indices rank-query for sorting
        query2rank = np.empty(ns)
        query2rank[rank2query] = range(ns)   # Inverted index q

        # Sort selection probabilities with query indices. Note the diference:
        #   rank2p_sel[k] is the selection probability fo the query with the
        #                 k-th highest score.
        #   p_sel[q] is the selection probability of query q.
        p_sel = np.array([rank2p_sel[int(r)] for r in query2rank])

        # Ranking for unlabeled data only
        queries_unl = [i for i in xrange(ns) if not islabeled[i]]
        scores_unl = [c_unl[q] for q in queries_unl]
        n_unl = len(queries_unl)
        rank_unl = np.argsort(scores_unl)
        rank_unl2query = [queries_unl[r] for r in rank_unl]

        # Ranking for labeled data only
        queries_lab = [i for i in xrange(ns) if islabeled[i]]
        scores_lab = [c_lab[q] for q in queries_lab]
        n_lab = len(queries_lab)
        rank_lab = np.argsort(scores_lab)
        rank_lab2query = [queries_lab[r] for r in rank_lab]

        # Report data sizes
        if self.sound == "on":
            print u"Número total de muestras: ", ns
            print u"    - No etiquetadas: ", n_unl
            print u"    - Etiquetadas: ", n_lab
            print u"Número de muestras a etiquetar: ", num_queries

        #########################
        # ## Active learning loop

        # Initialize records
        unl0_lab1 = [0] * num_queries
        rs0_al1 = [0] * num_queries
        query_pool = [0] * num_queries

        for n in range(num_queries):

            # Make a decision new label vs relabel
            is_relabel = np.random.uniform() < self.p_relabel
            # If there are no samples from the selected type, flip is_relabel
            is_relabel = ((is_relabel and n_lab > 0) or
                          (not is_relabel and n_unl == 0))
            unl0_lab1[n] = int(is_relabel)

            # Make a decision random sampling (rs) vs active learning (al)
            is_al = np.random.uniform() < self.p_al
            rs0_al1[n] = int(is_al)

            ########################################
            # ## Unlabelled samples. Random sampling
            if not is_relabel and not is_al:

                # Take out sample at random from the unlabeled set
                ind = np.random.randint(n_unl)
                newquery = rank_unl2query[ind]
                rank_unl2query = np.delete(rank_unl2query, ind)

            ######################################
            # ## Labelled samples. Random sampling
            elif is_relabel and not is_al:

                # Take out sample at random from the unlabeled set
                ind = np.random.randint(n_lab)
                newquery = rank_lab2query[ind]
                rank_lab2query = np.delete(rank_lab2query, ind)

            #######################################
            # ## Unlabeled samples. Active learning
            elif not is_relabel and is_al:

                # The following applies to both tourney and greedy AL
                # Select 'tourneysize' players at random for this tourney
                if self.name == "tourney":
                    ts = self.tourneysize   # This is just to abbreviate
                    n_ext = n_unl + ts - 1
                elif self.name == "greedy":
                    ts = n_unl
                    n_ext = n_unl

                # The TOURNEY: Take ts players at random, and select the winner
                iPlayers = np.random.permutation(range(n_ext))[0:ts]
                iWinner = min(iPlayers)
                newquery = rank_unl2query[iWinner]

                # Move the selected sample from the list of eligible
                # queries to the pool of selected queries
                rank_unl2query = np.delete(rank_unl2query, iWinner)

            #######################################
            # ## Labeled samples. Active learning
            elif is_relabel and is_al:

                # The following applies to both tourney and greedy AL
                # Select 'tourneysize' players at random for this tourney
                if self.name == "tourney":
                    ts = self.tourneysize
                    n_ext = n_lab + ts - 1
                elif self.name == "greedy":
                    ts = n_lab
                    n_ext = n_lab

                # The TOURNEY: Take ts players at random, and select the winner
                iPlayers = np.random.permutation(range(n_ext))[0:ts]
                iWinner = min(iPlayers)
                newquery = rank_lab2query[iWinner]

                # Move the selected sample from the list of eligible
                # queries to the pool of selected queries
                rank_lab2query = np.delete(rank_lab2query, iWinner)

            query_pool[n] = newquery

            # Compute probability of selecting an already labeled sample
            Pu = sum(p_sel[q] for q in xrange(ns) if not islabeled[q])

            # Modify all current weights
            w_new += 1.0 / (ns * Pu) * np.array(islabeled)

            # The following is a more self-explanatory but less efficient
            # version of the weight modification.
            # # Expected number of labeled samples befor an unlabeled one
            # Pl = sum(p_sel[q] for q in xrange(ns) if islabeled[q])
            # e_lab = Pl / Pu
            # # Distribute e_lab among the labeled data
            # pp_lab = [p_sel[q]/Pl if islabeled[q] else 0 for q in xrange(ns)]
            # d_lab = np.array([e_lab * p for p in pp_lab])
            # self.weights = self.weights + d_lab / (ns * p_sel)

            # Compute the weight for the new query
            w_new[newquery] = 1.0 / (ns * p_sel[newquery])

            # Mark (or remark) new query as labeled
            islabeled[newquery] = True

            # Update the size of the sets of available queries
            n_unl -= not is_relabel
            n_lab -= is_relabel

        if self.sound == "on":
            nq_lab = np.sum(unl0_lab1)
            print u"Número de etiquetas nuevas: ", len(unl0_lab1) - nq_lab
            print u"Número de re-etiquetas: ", nq_lab

        return query_pool, unl0_lab1, rs0_al1, w_new

    def get_queries_old(self, num_queries, labels, s):
        """
        Select new label queries applying and active learning algorithm

        It applies the active learning algorithm specified in self.name

        Args:
            num_queries: number of queries to be returned.
            labels : vector of current labels
                     = -1 or 1 if the label is known
                     = 0 if the label is unknown, and must be labeled
                     Any other value is processed as if 0
            s      : soft decision of the classifier

        Returns:
            query_pool :Vector of indices to the selected samples
            rs0_al1:    Vector indicating which samples have been selected
                        by random sampling (0) or by active learning (1)
            unl0_lab1:  Vector indicating (with 1) which samples already
                        had a label (0 if not)
            w_pool:    :Vector of selection probabilities. It containts, for
                        query in query_pool, the probability with wich it was
                        selected. Selection probabilities are useful to
                        compute unbiased average measures from the sample set
                        selected by the AL.
        """

        # Dataset size
        n = len(s)

        # The number of selected queries cannot exceed the dataset size.
        num_queries = min(num_queries, n)

        ############################
        # ## Scoring unlabelled data

        # Compute a 0-1 score (certainty measure of the class) per sample.
        c_unl = abs(np.array(s)-self.alth)
        c_unl = c_unl/max(c_unl + sys.float_info.min)   # Add eps to avoid 0/0

        # Possible unlabeled queries
        queries_unl = [i for i in xrange(n) if abs(labels[i]) != 1]
        scores_unl = [c_unl[q] for q in queries_unl]
        n_unl = len(queries_unl)

        rank_unl = np.argsort(scores_unl)
        sorted_queries_unl = [queries_unl[r] for r in rank_unl]

        ##########################
        # ## Scoring labelled data

        # The score is a discrepancy measure between label and soft decision
        c_lab = 1 - abs(np.array(s)-labels) / 2

        # Possible labeled queries
        queries_lab = [i for i in xrange(n) if abs(labels[i]) == 1]
        scores_lab = [c_lab[q] for q in queries_lab]
        n_lab = len(queries_lab)

        rank_lab = np.argsort(scores_lab)
        sorted_queries_lab = [queries_lab[r] for r in rank_lab]

        # Report data sizes
        if self.sound == "on":
            print u"Número total de muestras: ", n
            print u"    - No etiquetadas: ", n_unl
            print u"    - Etiquetadas: ", n_lab
            print u"Número de muestras a etiquetar: ", num_queries

        #########################
        # ## Active learning loop

        # Initialize records
        unl0_lab1 = [0] * num_queries
        rs0_al1 = [0] * num_queries
        query_pool = [0] * num_queries
        w_pool = [0] * num_queries

        for n in range(num_queries):

            # Make a decision new label vs relabel
            is_relabel = np.random.uniform() < self.p_relabel
            # If there are no samples from the selected type, flip is_relabel
            is_relabel = ((is_relabel and n_lab > 0) or
                          (not is_relabel and n_unl == 0))
            unl0_lab1[n] = int(is_relabel)

            # Make a decision random sampling (rs) vs active learning (al)
            is_al = np.random.uniform() < self.p_al
            rs0_al1[n] = int(is_al)

            ########################################
            # ## Unlabelled samples. Random sampling
            if not is_relabel and not is_al:

                # Take sample at random from the unlabeled set
                ind = np.random.randint(n_unl)
                newquery = sorted_queries_unl[ind]

                sorted_queries_unl = np.delete(sorted_queries_unl, ind)

                # Conditional selection probabilities for rs_unl samples.
                p_sel = (1.0 - self.p_relabel) * (1.0 - self.p_al) / n_unl

            ######################################
            # ## Labelled samples. Random sampling
            elif is_relabel and not is_al:

                # Take out sample at random from the unlabeled set
                ind = np.random.randint(n_lab)
                newquery = sorted_queries_lab[ind]

                sorted_queries_lab = np.delete(sorted_queries_lab, ind)

                # Selection probabilities for rs_lab samples.
                p_sel = self.p_relabel * (1.0 - self.p_al) / n_lab

            #######################################
            # ## Unlabeled samples. Active learning
            elif not is_relabel and is_al:

                # The following applies to both tourney and greedy AL
                # Select 'tourneysize' players at random for this tourney
                if self.name == "tourney":
                    ts = self.tourneysize   # This is just to abbreviate
                    n_ext = n_unl + ts - 1
                elif self.name == "greedy":
                    ts = n_unl
                    n_ext = n_unl

                iPlayers = np.random.permutation(range(n_ext))[0:ts]
                iWinner = min(iPlayers)
                newquery = sorted_queries_unl[iWinner]

                # Move the selected sample from the list of eligible
                # queries to the pool of selected queries
                sorted_queries_unl = np.delete(sorted_queries_unl, iWinner)

                # Conditional selection probabilities for al_unl samples.
                p_sel = ((1.0 - self.p_relabel) * self.p_al * ts *
                         self.tourney_prob(iWinner + 1, n_ext, ts))

            #######################################
            # ## Labeled samples. Active learning
            elif is_relabel and is_al:

                # The following applies to both tourney and greedy AL
                # Select 'tourneysize' players at random for this tourney
                if self.name == "tourney":
                    ts = self.tourneysize
                    n_ext = n_lab + ts - 1
                elif self.name == "greedy":
                    ts = n_lab
                    n_ext = n_lab
                iPlayers = np.random.permutation(range(n_ext))[0:ts]
                iWinner = min(iPlayers)
                newquery = sorted_queries_lab[iWinner]

                # Move the selected sample from the list of eligible
                # queries to the pool of selected queries
                sorted_queries_lab = np.delete(sorted_queries_lab, iWinner)

                # Conditional selection probabilities for al_lab samples.
                p_sel = (self.p_relabel * self.p_al * ts *
                         self.tourney_prob(iWinner + 1, n_ext, ts))

            query_pool[n] = newquery

            n_tot = (self.p_relabel > 0) * n_lab + (self.p_relabel < 1) * n_unl
            self.w = n_tot * self.w * p_sel
            w_pool[n] = self.w

            # if is_al:
            #     pvec = [self.tourney_prob(ii + 1, n_unl, ts) * ts
            #             for ii in range(n_unl - ts + 1)]
            #     print "n_tot = n_unl = ", n_unl
            #     print "self.w = ", self.w

            # Update the size of the sets of availabel queries
            n_unl -= not is_relabel
            n_lab -= is_relabel

        if self.sound == "on":
            nq_lab = np.sum(unl0_lab1)
            print u"Número de etiquetas nuevas: ", len(unl0_lab1) - nq_lab
            print u"Número de re-etiquetas: ", nq_lab

        return query_pool, unl0_lab1, rs0_al1, w_pool

    def evaluate(self, myClassifier, x, true_labels, pool_size, n_sim, th=0.5):
        """
        Evaluates an AL algoritm for given dataset and classifier model

        The dataset and the classifier must be entered as parameters. The
        method estimates the error rate as a function of the number of labeled
        samples.

        The algorithm averages results over a number of simlation with the same
        dataset. This is usefull if the classifier or the AL algorithm have any
        stochastic component.

        Args:

            myClassifier: Classifier object: it must contain two methods:
                          fit and predict_proba (returning a probability
                          estimate for each class). (E.g., you can try with
                          sklearn.linear_model.LogisticRegression)
            x           : Data samples, each row is a sample
            true_labels : Array of binary labels, one per sample
            pool_size   : Number of queries requested in each pool of the AL
            n_sim       : Number of simulations to average
            th(=0)      : Decision threshold used by the classifier model

        Returns:
            Pe          : Array of average error rates based on the whole
                          dataset, as a function of the number of samples.
            PeRaw       : Array of error rates estimated by unweighted
                          averaging over labeled samples
            PeW         : Array of error rates estimated by weighted averaging
                          over labeled samples
        """

        # ## Size of dataset
        n = x.shape[0]  # Sample size

        # ## Initialize aggregate results
        n_iter = int(math.ceil(float(n)/pool_size))

        Pe = np.zeros(n_iter)
        PeRaw = np.zeros(n_iter)
        PeW = np.zeros(n_iter)

        print '    Averaging {0} simulations. Estimated time...'.format(n_sim)

        # ## Loop over simulation runs
        for i in range(n_sim):

            ##############################
            # ## Classifier initialization
            Labels = [None] * n    # Vector of current labels
            s = [self.alth] * n         # Initial scores
            w = None            # Initial weights

            if i == 0:
                start = time.clock()

            ######################
            # Active learning loop
            TrueErrorRate = np.zeros(n_iter)
            RawErrorRate = np.zeros(n_iter)
            WErrorRate = np.zeros(n_iter)

            for k in range(n_iter):

                # Active learning iteration
                Queries, unl0_lab1, rs0_al1, w = self.get_queries(
                    pool_size, Labels, s, w)

                # Label selected samples
                for q in Queries:
                    Labels[q] = true_labels[q]

                # Currently labeled dataset
                iLabeled = [j for j in xrange(len(Labels)) 
                            if Labels[j] is not None]
                # iLabeled = np.nonzero(np.abs(Labels) == 1)[0]
                x_tr = x[iLabeled, ]
                y_tr = np.array([Labels[j] for j in iLabeled])
                w_tr = np.array([w[j] for j in iLabeled])

                # The following should be done by the classifier methods, but
                # it is not the case at the time of writing...
                # If all samples are from the same class, flip the label of one
                # I know, this makes no sense from the machine learning
                # perspective... It is just to avoid an error message.
                if len(np.unique(y_tr)) == 1:
                    y_tr[0] = (y_tr[0] + 1) % self.n_classes  

                # ########################
                # Ground truth evaluation:
                # First we train the classifier with all labels delivered by
                # the AL algorithm.
                myClassifier.fit(x_tr, y_tr)
                f = myClassifier.predict_proba(x)
                s = f[:, myClassifier.classes_[1] == 1]

                # Then, we evaluate this classifier with all labels
                # Note that training and test samples are being used in this
                # error rate. This could be refined in later versions, but it
                # is enough to check the behavior of the AL algorithms.
                # d = (np.array(s) >= th)
                d = np.argmax(f, axis=1)
                TrueErrorRate[k] = (float(np.count_nonzero(true_labels != d)) /
                                    n)

                # ##############
                # AL evaluations
                # Self evaluation.
                # First, we compute leave-one-out predictions
                n_k = len(y_tr)
                n_folds = min(10, n_k)
                count_all = [np.count_nonzero(y_tr==j) for j in
                             range(self.n_classes)]

                if len([j for j in count_all if j<n_folds])==0:
                    preds = skcv.cross_val_predict(myClassifier, x_tr, y_tr,
                                                   cv=n_folds, verbose=0)

                    # We estimate error rates with the AL labels, in 2 ways:
                    # 1: Raw errors: unweighted error count
                    RawErrorRate[k] = (float(np.count_nonzero(y_tr != preds)) /
                                       n_k)
                    # 2: Errors weignted by its inverse probability.
                    WErrorRate[k] = float(w_tr.dot(y_tr != preds))/np.sum(w_tr)

                else:
                    # We cannot evaluat using skcv, so we take error rates
                    # equal to the current averages.
                    RawErrorRate[k] = PeRaw[k]
                    WErrorRate[k] = PeW[k]

                # Show AL and Classifier messages only once
                if k == 0:
                    self.sound = 'off'

            # True error rates based on the whole dataset.
            Pe = (i*Pe + TrueErrorRate)/(i+1)
            # Error rate estimated by unweighted average over labeled samples
            PeRaw = (i*PeRaw + RawErrorRate)/(i+1)
            # Error rate estimated by weighted average over labeled samples
            PeW = (i*PeW + WErrorRate)/(i+1)

            if i == 0:
                tt = time.clock() - start
                print str(tt*n_sim) + ' segundos'

        return Pe, PeRaw, PeW
