import numpy as np
from sklearn.datasets import make_classification, make_blobs
from experiments.data import make_weak_true_partition
from wlc.WLweakener import computeM, weak_to_index
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from experiments.models import create_model, MyKerasClassifier
import inspect

import matplotlib.pyplot as plt

n_classes = 10
n_features = 100
n_samples = 10000
true_size = 0.1
prop_test = 0.9 # Proportion of true size to use as test
random_state = 0
np.random.seed(random_state)

#==============================================================#
# Create synthetic clean dataset
#==============================================================#
#n_redundant = 0
#n_clusters_per_class = 2
#n_informative = n_features
#X, y = make_classification(n_samples=n_samples, n_features=n_features,
#                           n_classes=n_classes, random_state=random_state,
#                           n_redundant=n_redundant,
#                           n_informative=n_informative,
#                           n_clusters_per_class=n_clusters_per_class)
#
centers = np.random.rand(n_classes, n_features)*2.0
cluster_std = np.abs(np.random.randn(n_classes)*2.0)
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                  cluster_std=cluster_std, random_state=random_state)
#X *= np.random.rand(n_features)*10.0
Y = np.zeros((y.size, y.max()+1))
Y[np.arange(n_samples), y] = 1

#==============================================================#
# Create synthetic Mixing process
#==============================================================#
method='random_weak'
alpha = 0.5
beta = 0.2
M = computeM(n_classes, method=method, alpha=alpha, beta=beta,
             seed=random_state)

#==============================================================#
# Create synthetic Weak labels given M
#==============================================================#
training, validation, test, classes = make_weak_true_partition(M, X, y,
                                                               true_size=true_size,
                                                               random_state=random_state)

X_t, Z_t, z_t = training
X_v, Z_v, z_v, Y_v, y_v = validation

sss = StratifiedShuffleSplit(n_splits=1, random_state=random_state,
                             train_size=(1. - prop_test),
                             test_size=prop_test)

val_indx, test_indx = next(sss.split(X_v, y_v))
print('Weak labels: Training original partition size = {}'.format(len(z_t)))
print('True labels: Validation original partition size = {}'.format(len(val_indx)))
print('True labels: Test original partition size = {}'.format(len(test_indx)))
# test partition
X_te, Z_te, z_te = X_v[test_indx], Z_v[test_indx], z_v[test_indx]
Y_te, y_te = Y_v[test_indx], y_v[test_indx]
# Validation partition
X_v, Z_v, z_v = X_v[val_indx], Z_v[val_indx], z_v[val_indx]
Y_v, y_v = Y_v[val_indx], y_v[val_indx]
print('True labels: Validation partition size = {}'.format(len(y_v)))
print('True labels: Test partition size = {}'.format(len(y_te)))

#==============================================================#
# Train Scikit learn baselines
#==============================================================#
LR = LogisticRegression()
LR.fit(X, y)
print('A Logistic Regression trained with all the real labels ({} samples)'.format(y.shape[0]))
acc_upperbound = LR.score(X_te, y_te)
print('Accuracy = {}'.format(acc_upperbound))

LR = LogisticRegression()
LR.fit(X_v, y_v)
print('A Logistic Regression trained with only validation true labels ({} samples)'.format(y_v.shape[0]))
acc_lowerbound = LR.score(X_te, y_te)
print('Accuracy = {}'.format(acc_lowerbound))

#==============================================================#
# set Neural Network parameters
#==============================================================#
params = {'input_dim': n_features,
          'output_size': n_classes,
          'optimizer': 'sgd',
          'loss': 'log_loss', #'mean_squared_error', #'log_loss'
#          'init': 'glorot_uniform',
#          'lr': 0.1,
#          'l1': 0.1,
#          'l2': 0.1,
#          'momentum': True,
          'decay': 0.1,
#          'rho': 0.1,
#          'epsilon': 0.1,
#          'nesterov': False,
          'epochs': 100,
#          'batch_size': 25,
          'verbose': 0,
          'random_state': random_state,
          'training_method': 'EM',
          'architecture': 'lr',
#          'path_model': None
          'model_num': 0
          }
fit_arguments = {key: value for key, value in params.items()
                 if key in inspect.getargspec(create_model().fit)[0]}
make_arguments = {key: value for key, value in params.items()
                  if key in inspect.getargspec(create_model)[0]}

#==============================================================#
# Train Keras baselines
#==============================================================#
method = 'supervised'
make_arguments['training_method'] = method
classifier = MyKerasClassifier(build_fn=create_model, **make_arguments)
# This fails with random noise (in that case the matrix M is not DxC but CxC)
classifier.fit(X, Y, **fit_arguments)
# 5. Evaluate the model in the validation set with true labels
y_pred = classifier.predict(X_te)
# Compute the confusion matrix
cm = confusion_matrix(y_te, y_pred)
print('cm:\n{}'.format(cm))
acc_keras_upperbound = cm.diagonal().sum()/cm.sum()
print('Accuracy = {}'.format(acc_keras_upperbound))

classifier = MyKerasClassifier(build_fn=create_model, **make_arguments)
Y_v = np.zeros((y_v.size, y_v.max()+1))
Y_v[np.arange(len(y_v)), y_v] = 1
classifier.fit(X_v, Y_v, **fit_arguments)
y_pred = classifier.predict(X_te)
cm = confusion_matrix(y_te, y_pred)
print('cm:\n{}'.format(cm))
acc_keras_lowerbound = cm.diagonal().sum()/cm.sum()
print('Accuracy = {}'.format(acc_keras_lowerbound))

#==============================================================#
# Train All methods
#==============================================================#
acc = {}
#list_weak_proportions = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.07, 0.1, 0.3,
#                                 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
list_weak_proportions = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0])
#==============================================================#
# Train EM with known M and with not known M
#==============================================================#
method = 'EM'
make_arguments['training_method'] = method
#for method, M_0 in zip(['EM M known', 'EM M unk.'],
#                       [M, np.ones_like(M)/M.shape[0]]):
for method, M_0 in zip(['EM M known'], [M]):
    # 1. Learn a mixing matrix using training with weak and true labels
    # In this code we give the true matrix M
    M_1 = computeM(c=n_classes, method='supervised')
    q_0 = X_t.shape[0] / float(X_t.shape[0] + X_v.shape[0])
    q_1 = X_v.shape[0] / float(X_t.shape[0] + X_v.shape[0])
    M_EM = np.concatenate((q_0*M_0, q_1*M_1), axis=0)
    #  2. Compute the index of each sample relating it to the corresponding
    #     row of the new mixing matrix
    #      - Needs to compute the individual M and their weight q
    Z_t_index = weak_to_index(Z_t, method='Mproper')
    Y_v_index = weak_to_index(Y_v, method='supervised')

    print('q0 = {}, q1 = {}'.format(q_0, q_1))
    print("M_0\n{}".format(np.round(M_0, decimals=3)))
    print("M_1\n{}".format(np.round(M_1, decimals=3)))
    print("M_EM\n{}".format(np.round(M_EM, decimals=3)))
    print("Z_t\n{}".format(np.round(Z_t[:5])))
    print("Z_t_index {}".format(Z_t_index[:5]))
    print('Y_v_index {}'.format(Y_v_index[:5]))
    print("Y_v\n{}".format(np.round(Y_v[:5])))

    # 3. Give the mixing matrix to the model for future use
    #    I need to give the matrix M to the fit function
    # 4. Train model using all the sets with instead of labels the index of
    #    the corresponding rows of the mixing matrix
    acc[method] = np.zeros_like(list_weak_proportions)
    for i, weak_proportion in enumerate(list_weak_proportions):
        last_index = int(weak_proportion*Z_t_index.shape[0])
        print('Number of weak samples = {}'.format(last_index))

        Z_index_t = np.concatenate((Z_t_index[:last_index], Y_v_index + M_0.shape[0]))
        np.random.seed(random_state)

        X_tv = np.concatenate((X_t[:last_index], X_v), axis=0)
        X_tv, Z_index_tv = shuffle(X_tv, Z_index_t)

        classifier = MyKerasClassifier(build_fn=create_model,
                                       **make_arguments)

        # This fails with random noise (in that case the matrix M is not DxC but CxC)
        classifier.fit(X_tv, Z_index_tv, M=M_EM, X_y_t=X_v, Y_y_t=Y_v,
                                 **fit_arguments)
        # 5. Evaluate the model in the validation set with true labels
        y_pred = classifier.predict(X_te)
        # Compute the confusion matrix
        cm = confusion_matrix(y_te, y_pred)
        print('cm:\n{}'.format(cm))
        acc[method][i] = cm.diagonal().sum()/cm.sum()
        print('Accuracy = {}'.format(acc[method][i]))

#==============================================================#
# Train weak and OSL
#==============================================================#
for method in ['weak', 'OSL']:
    make_arguments['training_method'] = method
    acc[method] = np.zeros_like(list_weak_proportions)
    for i, weak_proportion in enumerate(list_weak_proportions):
        last_index = int(weak_proportion*Z_t.shape[0])
        print('Number of weak samples = {}'.format(last_index))

        train_x = np.concatenate((X_t[:last_index], X_v), axis=0)
        train_y = np.concatenate((Z_t[:last_index], Y_v), axis=0)

        classifier = MyKerasClassifier(build_fn=create_model,
                                       **make_arguments)

        # This fails with random noise (in that case the matrix M is not DxC but CxC)
        classifier.fit(train_x, train_y, **fit_arguments)
        # 5. Evaluate the model in the validation set with true labels
        y_pred = classifier.predict(X_te)
        # Compute the confusion matrix
        cm = confusion_matrix(y_te, y_pred)
        print('cm:\n{}'.format(cm))
        acc[method][i] = cm.diagonal().sum()/cm.sum()
        print('Accuracy = {}'.format(acc[method][i]))

#==============================================================#
# Generate results
#==============================================================#
print('Acc. Upperbound = {}'.format(acc_upperbound))
for method in ['weak', 'OSL']:
    print('Acc. {}\n{}'.format(method, acc[method]))
print('Acc. Lowerbound = {}'.format(acc_lowerbound))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Accuracy on test set with {} true labels'.format(X_te.shape[0]))
for color, method in zip(['b', 'c', 'g', 'purple'],
                         ['EM M known', 'EM M unk.', 'weak', 'OSL']):
    if method in acc.keys():
        ax.plot(list_weak_proportions*Z_t.shape[0], acc[method], 'o-',
                color=color, label='{} weak + {} true labels'.format(method,
                                                                     Z_v.shape[0]))
ax.axhline(y=acc_upperbound, color='r', linestyle='-', label='{} true labels'.format(X.shape[0]))
ax.axhline(y=acc_keras_upperbound, color='pink', linestyle='-', label='{} Keras true labels'.format(X.shape[0]))
ax.axhline(y=acc_lowerbound, color='orange', linestyle='-', label='{} true labels'.format(Z_v.shape[0]))
ax.axhline(y=acc_keras_lowerbound, color='yellow', linestyle='-', label='{} Keras true labels'.format(Z_v.shape[0]))
ax.set_xlabel('Number of weak samples')
ax.set_ylabel('Accuracy')
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
art = [lgd]
#fig.tight_layout()
fig.savefig('full_vs_others.svg', additional_artists=art, bbox_inches="tight")
