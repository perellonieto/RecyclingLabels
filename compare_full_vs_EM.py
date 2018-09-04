import numpy as np
from sklearn.datasets import make_classification
from experiments.data import make_weak_true_partition
from wlc.WLweakener import computeM, weak_to_index
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from experiments.models import create_model, MyKerasClassifier
import inspect

import matplotlib.pyplot as plt

n_classes = 6
n_features = 10
n_informative = n_features
n_redundant = 0
n_samples = 10000
true_size = 0.1
n_clusters_per_class = 2
random_state = 0

#==============================================================#
# Create synthetic clean dataset
#==============================================================#
X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_classes=n_classes, random_state=random_state,
                           n_redundant=n_redundant,
                           n_informative=n_informative,
                           n_clusters_per_class=n_clusters_per_class)

#==============================================================#
# Create synthetic Mixing process
#==============================================================#
method='random_weak'
alpha = 0.5
beta = 0.3
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

prop_test = 0.9
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
print('A Logistic Regression trained with all the real labels')
acc_upperbound = LR.score(X_te, y_te)
print('Accuracy = {}'.format(acc_upperbound))

LR = LogisticRegression()
LR.fit(X_v, y_v)
print('A Logistic Regression trained with only validation true labels')
acc_lowerbound = LR.score(X_te, y_te)
print('Accuracy = {}'.format(acc_lowerbound))

#==============================================================#
# Train EM
#==============================================================#
process_id = 0
classifier = 'lr'

categories = range(n_classes)
# 1. Learn a mixing matrix using training with weak and true labels
# In this code we give the true matrix M
M_0 = M
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
list_weak_proportions = np.array([0, 0.01, 0.02, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0])
acc_EM = np.zeros_like(list_weak_proportions)
for i, weak_proportion in enumerate(list_weak_proportions):
    last_index = int(weak_proportion*Z_t_index.shape[0])
    print('Number of weak samples = {}'.format(last_index))

    Z_index_t = np.concatenate((Z_t_index[:last_index], Y_v_index + M_0.shape[0]))
    np.random.seed(process_id)

    X_tv = np.concatenate((X_t[:last_index], X_v), axis=0)
    X_tv, Z_index_tv = shuffle(X_tv, Z_index_t)

    # Add validation results during training
    # Int his example we do not have validation
    #fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    params = {'input_dim': n_features,
              'output_size': n_classes,
              'optimizer': 'sgd',
              'loss': 'mean_squared_error',
    #          'init': 'glorot_uniform',
    #          'lr': 0.1,
    #          'l1': 0.1,
    #          'l2': 0.1,
    #          'momentum': True,
    #          'decay': 0.1,
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
              }
    fit_arguments = {key: value for key, value in params.items()
                     if key in inspect.getargspec(create_model().fit)[0]}
    make_arguments = {key: value for key, value in params.items()
                      if key in inspect.getargspec(create_model)[0]}
    make_arguments['model_num'] = process_id
    classifier = MyKerasClassifier(build_fn=create_model,
                                   **make_arguments)

    # This fails with random noise (in that case the matrix M is not DxC but CxC)
    history = classifier.fit(X_tv, Z_index_tv, M=M_EM, X_y_t=X_v, Y_y_t=Y_v,
                             **fit_arguments)
    # 5. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_te)
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_te, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history.history}
    print('cm:\n{}'.format(cm))
    acc_EM[i] = cm.diagonal().sum()/cm.sum()
    print('Accuracy = {}'.format(acc_EM[i]))

print('Acc. Upperbound = {}'.format(acc_upperbound))
print('Acc. EM\n{}'.format(acc_EM))
print('Acc. Lowerbound = {}'.format(acc_lowerbound))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(list_weak_proportions*Z_t_index.shape[0], acc_EM, 'bo-', label='EM weak + {} true labels'.format(Z_v.shape[0]))
ax.axhline(y=acc_upperbound, color='r', linestyle='-', label='{} true labels'.format(X.shape[0]))
ax.axhline(y=acc_lowerbound, color='orange', linestyle='-', label='{} true labels'.format(Z_v.shape[0]))
ax.set_xlabel('Number of weak samples')
ax.set_ylabel('Accuracy')
ax.legend(loc=0)
fig.savefig('full_vs_EM.svg')
