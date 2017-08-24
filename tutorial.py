import sys

import numpy as np
from experiments.data import load_webs, load_weak_iris, load_weak_blobs
from experiments.metrics import compute_expected_error, compute_error_matrix

from wlc.WLweakener import computeM, estimate_M, weak_to_index
from wlc.WLweakener import generateWeak
from wlc.WLweakener import binarizeWeakLabels

from sklearn.preprocessing import label_binarize

print("Most of the following floats are limited to a precision of 2 decimals")
np.set_printoptions(precision=2)

def brier_score(q, p):
    q = np.array(q)
    p = np.array(p)
    return np.sum((q-p)**2)


def log_loss(q, p):
    q = np.array(q)
    p = np.array(p)
    return -(p*np.log(q)).sum()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def gradSquareLossEM(w, X, T):
    n_dim = X.shape[1]
    p = np.array(map(softmax, np.dot(X, w)))
    Q = (p - T) * p
    sumQ = np.sum(Q, axis=1, keepdims=True)
    G = np.dot(X.T, Q - sumQ * p)
    return G


def print_code(text):
    print('\n<pre>')
    print(text)
    print('</pre>\n')


load_dataset = {'blobs': load_weak_blobs,
                'iris': load_weak_iris,
                'webs': load_webs}

if sys.argv > 1:
    if str(sys.argv[1]) not in load_dataset:
        print("Wrong argument: Optional datasets are")
        print(load_dataset.keys())
        sys.exit()
    dataset = str(sys.argv[1])
else:
    dataset = 'iris'
n_sv = 6 # Number of samples to visualise
mixing_method = "quasi_IPL"
alpha = 0.5
beta = 0.3
iterate_test = True
iterations = 20
simulate = ['GD', 'EM'] # GD, EM
print("\n# DATA DESCRIPTION ###")
print("\n- name = " + dataset)
training, validation, classes = load_dataset[dataset](random_state=0)
X_t, Z_t, z_t = training
X_v, Z_v, z_v, Y_v, y_v = validation
n_f = X_t.shape[1]
print("- Number of features = {}".format(n_f))
n_c = Y_v.shape[1]
print("- Number of classes = {}".format(n_c))
print("- Classes = {}".format(classes))
print("- Samples with only weak = {}".format(len(z_t)))
print("- Samples with true labels = {}".format(len(y_v)))
if dataset in ['iris', 'blobs']:
    print("\n## HOW THE WEAK LABELS ARE GENERATED ###")
    print('''\nThis dataset only has true labels. In order to create the weak
            labels we designed a mixing matrix with the following steps:''')
    print("\nGiven the decimal true labels y")
    # We will use subindex _s to denote the small sample for visualisation
    y_s = y_v[:n_sv]
    print("\n- y = {}".format(y_s))
    print("\nWe create artificially the mixing matrix M")
    print("\n- method = {}\n- alpha = {}\n- beta = {}".format(mixing_method, alpha, beta))
    print("\nResulting mixing matrix M")
    M = computeM(n_c, method=mixing_method, alpha=alpha, beta=beta)
    print_code(M)
    print("\nBinarize the true labels and store in Y")
    Y_s = label_binarize(y_s, classes)
    print_code(Y_s)
    print("\nGenerate the weak labels with the mixing matrix M")
    z_s = generateWeak(y_s, M, seed=0)
    print_code(z_s)
    print("\nBinarize the weak labels and store in Z")
    Z_s = binarizeWeakLabels(z_s, c=n_c)
    print_code(Z_s)

print("\n## DATA SAMPLE ###")
# Subindex _s is used to denote the samples for visualisation
X_s, Z_s, z_s, Y_s, y_s = X_v[:n_sv], Z_v[:n_sv], z_v[:n_sv], Y_v[:n_sv], y_v[:n_sv]
print("\nExample of true labels Y")
print("\n- In decimal y =")
print_code(y_s)
print("\n- In binary Y =")
print_code(Y_s)
print("\nExample of corresponding weak labels Z")
print("\n- In decimal z =")
print_code(z_s)
print("\n- In binary Z =")
print_code(Z_s)
print("\nPrior distribution for **all** true labels. $P(Y)$")
prior_y = np.true_divide(Y_v.sum(axis=0), Y_v.sum())
print_code(prior_y)

print("\n# EXPECTED ERROR FOR A BASELINE ###")
print("\nWe show the performance of a simple model that always predicts the prior")
print("\n## BRIER SCORE #")
print("Error matrix with Brier score: $\Psi_{BS} = BS(P(Y), I)$")
bs_matrix = compute_error_matrix(prior_y, brier_score)
print_code(bs_matrix)
print("\nExpected Brier score: $\mathbb{E}_{y\sim P(y)} [\Psi_{BS}(S, y)] = \sum_{j=1}^K P(y=j) \Psi_{BS}(S, y_j)$")
expected_bs = compute_expected_error(prior_y, bs_matrix)
print_code(expected_bs)

print("\n## LOG-LOSS #")
print("Error matrix with Log-loss: $\Psi_{LL} = LL(P(Y), I)$")
ll_matrix = compute_error_matrix(prior_y, log_loss)
print_code(ll_matrix)
print("\nExpected Log-loss: $\mathbb{E}_{y\sim P(y)} [\Psi_{LL}(S, y)] = \sum_{j=1}^K P(y=j) \Psi_{LL}(S, y_j)$")
expected_ll = compute_expected_error(prior_y, ll_matrix)
print_code(expected_ll)

print("\n# ESTIMATION OF THE MIXING MATRIX M ###")
print("From now, $M_0$ is for the weak set and $M_1$ for the **full set** that contains the true labels")
## 1. Learn a mixing matrix using training with weak and true labels
print("\nEstimated $M_0$ without Laplace correction")
M_0 = estimate_M(Z_v, Y_v, classes, reg=None)
print_code(M_0)
print("\nEstimated $M_0$ with Laplace correction")
M_0 = estimate_M(Z_v, Y_v, classes, reg='Complete')
print_code(M_0)
print("\nThe mixing matrix for the clean data $M_1$")
M_1 = computeM(c=n_c, method='supervised')
print_code(M_1)

print("\n## COMBINATION OF BOTH MATRICES FOR THE FULL DATASET ###")
print("\nProportion of samples with weak ($q_0$) and with true labels ($q_1$)\n")
q_0 = len(z_t) / float(len(z_t) + len(y_v))
print("- $q_0$ = {}".format(q_0))
q_1 = len(y_v) / float(len(z_t) + len(y_v))
print("- $q_1$ = {}".format(q_1))
print("\nComposition of mixing matrices $M = [q_0*M_0 \mathtt{ , } q_1*M_1]^T$")
M = np.concatenate((q_0*M_0, q_1*M_1), axis=0)
print_code(M)
print("\nThe corresponding indices of the weak labels to the rows of the matrix M")
Z_s_index = weak_to_index(Z_s, method='Mproper')
print_code(Z_s_index)
print("\nThe corresponding indices of the true labels to the rows of the matrix M")
Y_s_index = weak_to_index(Y_s, method='supervised') + M_0.shape[0]
print_code(Y_s_index)

print("\n# TRAINING SOME MODELS ###")
if X_s.shape[1] == n_f:
    print("\nFrom here I will add an extra feature fixed to 1 for the bias")
    X_s = np.c_[X_s, np.ones(X_s.shape[0])]
    print_code(X_s)
if X_v.shape[1] == n_f:
    X_v = np.c_[X_v, np.ones(X_v.shape[0])]

if 'GD' in simulate:
    print("\n## SIMULATION OF GRADIENT DESCENT FOR A LR ###")
    print("\nInitial weights w")
    np.random.seed(0)
    m_W = np.random.randn(X_s.shape[1], n_c)
    print_code(m_W)
    print("\nThe small sample of X")
    print_code(X_s)
    print("\nInitial model activations m_a = X m_W")
    m_a = np.dot(X_s,m_W)
    print_code(m_a)
    print("\nInitial model outputs m_q = softmax(m_a)")
    m_q = np.array(map(softmax, m_a))
    print_code(m_q)
    print("\nInitial model predictions m_pred = argmax(m_q)")
    m_pred = np.argmax(m_q, axis=1)
    print_code(m_pred)
    print("\nCompute the error of the predicted probabilities q true labels Y")
    error = np.array([brier_score(q_i, p_i) for q_i, p_i in zip(m_q, Y_s)])
    print_code(error)
    print("\nCompute the gradient of the Brier score")
    G = np.dot(X_s.T, m_q - Y_s)
    print_code(G)
    print("\nUpdate the weights $W_{t+1} = W_t - G$")
    m_W -= G
    print_code(m_W)

    print("\n### WE CAN PERFORM GD FOR SOME ITERATIONS FULL TRUE SET ###")
    if iterate_test:
        print("\nIterations :\n")
        print("| it | mean bs | acc |")
        print("| :--- | :-----------: | :-----: |")
        for i in range(iterations):
            m_a = np.dot(X_v,m_W)
            m_q = np.array(map(softmax, m_a))
            m_pred = np.argmax(m_q, axis=1)
            error = np.array([brier_score(q_i, p_i) for q_i, p_i in zip(m_q, Y_v)])
            print("| {0} | {1:.2E} | {2:.2f} |".format(i, error.mean(),
                                                       np.mean(y_v == m_pred)))
            m_G = np.dot(X_v.T, m_q - Y_v)
            m_W -= m_G

if 'EM' in simulate:
    print("\n## SIMULATION OF THE EXPECTATION MAXIMIZATION STEPS FOR A LR ###")
    print("\nInitial weights w")
    np.random.seed(0)
    m_W = np.random.randn(X_s.shape[1], n_c)
    print_code(m_W)
    print("\nThe small sample of X")
    print_code(X_s)
    print("\n#### EXPECTATION STEP: Assume the weights of the model are right ###")
    print("\nInitial prediction Z = X w")
    m_a = np.dot(X_s,m_W)
    print_code(m_a)
    print("\nInitial q = softmax(Z)")
    m_q = np.array(map(softmax, m_a))
    print_code(m_q)
    print("\nInitial predictions")
    m_pred = np.argmax(m_q, axis=1)
    print_code(m_pred)
    print("\nCompute the virtual labels Q = q * M[Z_index]")
    m_Q = np.multiply(m_q, M[Z_s_index])
    print_code(m_Q)
    print("\nNormalize the virtual labels to sum to one")
    m_Q = m_Q / np.sum(m_Q, axis=1)
    print_code(m_Q)
    print("\nSearch for infinite or NaN values and remove them for this training step")
    print("The following samples are NOT removed from the training")
    fin_ind = np.where(np.isfinite(np.sum(m_Q, axis=1)))[0]
    print_code(fin_ind)
    print("\n#### MAXIMIZATION STEP: In our case update the weights to MINIMIZE the error###")
    print("\nCompute the error of the predicted probabilities q against the new Virtual labels Q")
    error = np.array([brier_score(q_i, p_i) for q_i, p_i in zip(m_q[fin_ind], m_Q[fin_ind])])
    print_code(error)
    print("\nCompute the gradient of the Brier score")
    m_G = gradSquareLossEM(m_W, X_s, np.array(m_Q))
    print_code(m_G)
    print("\nUpdate the weights $W_{t+1} = W_t - G$")
    m_W -= m_G
    print_code(m_W)

    print("\n### WE CAN PERFORM EM FOR SOME ITERATIONS ###")
    Z_v_index = weak_to_index(Z_v, method='Mproper')
    print("\nIterations :\n")
    print("| it | mean bs | acc |")
    print("| :-- | :-------: | :---: |")
    for i in range(iterations):
        # Expectation
        m_a = np.dot(X_v,m_W)
        m_q = np.array(map(softmax, m_a))
        m_pred = np.argmax(m_q, axis=1)
        m_Q = np.multiply(m_q, M[Z_v_index])
        m_Q = m_Q / np.sum(m_Q, axis=1)
        fin_ind = np.where(np.isfinite(np.sum(m_Q, axis=1)))[0]
        error = np.array([brier_score(q_i, p_i) for q_i, p_i in zip(m_q[fin_ind], m_Q[fin_ind])])
        print("| {0} | {1:.2E} | {2:.2f}".format(i, error.mean(),
                                                 np.mean(y_v == m_pred)))
        # Maximization
        m_G = gradSquareLossEM(m_W, X_v[fin_ind], np.array(m_Q)[fin_ind])
        m_W -= m_G
