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


load_dataset = {'blobs': load_weak_blobs,
                'iris': load_weak_iris,
                'webs': load_webs}

dataset = 'iris'
sv = 6 # Number of samples to visualise
mixing_method = "quasi_IPL"
alpha = 0.5
beta = 0.3
print("\n### DATA DESCRIPTION ###")
print("name = " + dataset)
training, validation, classes = load_dataset[dataset](random_state=0)
X_t, Z_t, z_t = training
X_v, Z_v, z_v, Y_v, y_v = validation
n_f = X_t.shape[1]
print("Number of features = {}".format(n_f))
n_c = Y_v.shape[1]
print("Number of classes = {}".format(n_c))
print("Classes = {}".format(classes))
print("Samples with only weak = {}".format(len(z_t)))
print("Samples with true labels = {}".format(len(y_v)))
if dataset in ['iris', 'blobs']:
    print("\n### HOW THE WEAK LABELS ARE GENERATED ###")
    print("\nIn this dataset we created the weak labels in the following manner")
    print("\nGiven the decimal true labels y")
    y = y_v[:sv]
    print("y = {}".format(y))
    print("\nWe create artificially the mixing matrix M")
    print("\nmethod = {}, alpha = {}, beta = {}".format(mixing_method, alpha, beta))
    print("\nResulting mixing matrix M")
    M = computeM(n_c, method=mixing_method, alpha=alpha, beta=beta)
    print(M)
    print("\nBinarize the true labels and store in Y")
    Y = label_binarize(y, classes)
    print(Y)
    print("\nGenerate the weak labels with the mixing matrix M")
    z = generateWeak(y, M, seed=0)
    print(z)
    print("\nBinarize the weak labels and store in Z")
    Z = binarizeWeakLabels(z, c=n_c)
    print(Z)

print("\n### DATA SAMPLE ###")
X, Z, z, Y, y = X_v[:sv], Z_v[:sv], z_v[:sv], Y_v[:sv], y_v[:sv]
print("\nExample of true labels Y")
print("- In decimal y =")
print(y)
print("- In binary Y =")
print(Y)
print("\nExample of corresponding weak labels Z")
print("- In decimal z =")
print(z)
print("- In binary Z =")
print(Z)
print("\nPrior distribution for all true labels. p_y")
prior_y = np.true_divide(Y_v.sum(axis=0), Y_v.sum())
print(prior_y)

print("\n### EXPECTED ERROR FOR A BASELINE ###")
print("\n# BRIER SCORE #")
print("Error matrix with Brier score for a model that always predicts the prior")
bs_matrix = compute_error_matrix(prior_y, brier_score)
print(bs_matrix)
print("\nExpected Brier score for the always prior model")
expected_bs = compute_expected_error(prior_y, bs_matrix)
print(expected_bs)

print("\n# LOG-LOSS #")
print("Error matrix with Log-loss for a model that always predicts the prior")
ll_matrix = compute_error_matrix(prior_y, log_loss)
print(ll_matrix)
print("\nExpected Log-loss for the always prior model")
expected_ll = compute_expected_error(prior_y, ll_matrix)
print(expected_ll)


print("\n### ESTIMATION OF THE MIXING MATRIX M ###")
print("From now, M_0 is for the weak set and M_1 for the set with true labels")
## 1. Learn a mixing matrix using training with weak and true labels
print("\nEstimated M_0 without Laplace correction")
M_0 = estimate_M(Z_v, Y_v, classes, reg=None)
print(M_0)
print("\nEstimated M_0 with Laplace correction")
M_0 = estimate_M(Z_v, Y_v, classes, reg='Complete')
print(M_0)
print("\nThe mixing matrix for the clean data M_1")
M_1 = computeM(c=n_c, method='supervised')
print(M_1)

print("\n### COMBINATION OF BOTH MATRICES FOR EXPECTATION MAXIMIZATION ###")
print("\nProportions of samples with weak (q_0) and with true labels (q_1)")
q_0 = len(z_t) / float(len(z_t) + len(y_v))
print("q_0 = {}".format(q_0))
q_1 = len(y_v) / float(len(z_t) + len(y_v))
print("q_1 = {}".format(q_1))
print("\nComposition of mixing matrices q_0*M_0 and q_1*M_1")
M = np.concatenate((q_0*M_0, q_1*M_1), axis=0)
print(M)
print("\nThe corresponding indices of the weak labels to the rows of the matrix M")
Z_index = weak_to_index(Z, method='Mproper')
print(Z_index)
print("\nThe corresponding indices of the true labels to the rows of the matrix M")
Y_index = weak_to_index(Y, method='supervised') + M_0.shape[0]
print(Y_index)

print("\n### SIMULATION OF THE EXPECTATION MAXIMIZATION STEPS FOR A LR ###")
if X.shape[1] == n_f:
    print("\nFrom here I will add an extra feature fixed to 1 for the bias")
    X = np.c_[X, np.ones(X.shape[0])]
    print(X)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
print("\nInitial weights w")
np.random.seed(0)
W = np.random.randn(X.shape[1], n_c)
print(W)
print("\nThe small sample of X")
print(X)
print("\n### EXPECTATION STEP: Assume the weights of the model are right ###")
print("\nInitial prediction Z = X·w")
q_z = np.dot(X,W)
print(z)
print("\nInitial q = softmax(Z)")
q = np.array(map(softmax, q_z))
print(q)
print("\nInitial predictions")
pred = np.argmax(q, axis=1)
print(pred)
print("\nCompute the virtual labels Q = q * M[Z_index]")
Q = np.multiply(q, M[Z_index])
print(Q)
print("\nNormalize the virtual labels to sum to one")
Q = Q / np.sum(Q, axis=1)
print(Q)
print("\nSearch for infinite or NaN values and remove them for this training step")
print("The following samples are NOT removed from the training")
fin_ind = np.where(np.isfinite(np.sum(Q, axis=1)))[0]
print(fin_ind)
print("\n### MAXIMIZATION STEP: In our case update the weights to MINIMIZE the error###")
print("\nCompute the error of the predicted probabilities q against the new Virtual labels Q")
error = np.array([brier_score(q_i, p_i) for q_i, p_i in zip(q[fin_ind], Q[fin_ind])])
print(error)
print("\nCompute the gradient of the Brier score")
def gradSquareLoss(w, X, T):
    n_dim = X.shape[1]
    p = np.array(map(softmax, np.dot(X, w)))
    Q = (p - T) * p
    sumQ = np.sum(Q, axis=1, keepdims=True)
    G = np.dot(X.T, Q - sumQ * p)
    return G
G = gradSquareLoss(W, X, np.array(Q))
print(G)
print("\nUpdate the weights W_t+1 = W_t - G")
W -= G
print(W)

print("\n### WE CAN PERFORM EM FOR SOME ITERATIONS ###")
for i in range(10):
    # Expectation
    q_z = np.dot(X,W)
    q = np.array(map(softmax, q_z))
    pred = np.argmax(q, axis=1)
    Q = np.multiply(q, M[Z_index])
    Q = Q / np.sum(Q, axis=1)
    fin_ind = np.where(np.isfinite(np.sum(Q, axis=1)))[0]
    error = np.array([brier_score(q_i, p_i) for q_i, p_i in zip(q[fin_ind], Q[fin_ind])])
    print("[{0}] mean_BS = {1:.2E}, Acc. = {2}".format(i, error.mean(),
                                                       np.mean(y == pred)))
    # Maximization
    G = gradSquareLoss(W, X[fin_ind], np.array(Q)[fin_ind])
    W -= G
