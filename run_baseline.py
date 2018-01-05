import argparse
import inspect

import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from experiments.data import load_webs, load_weak_iris, load_weak_blobs, \
                             load_toy_example, load_classification
from experiments.analysis import analyse_weak_labels
from experiments.diary import Diary

DEFAULT = {'dataset': 'iris',
           'seed': 42,
           'verbose': 0,
           'epochs': 200,
           'lr': 1.0,
           'l1': 0.0,
           'l2': 0.001,
           'architecture': 'lr',
           'optimizer': 'rmsprop',
           'method': 'fully_supervised',
           'n_jobs': None,
           'n_iterations': 2,
           'k_folds': 2,
           'path_results': 'results',
           'loss': 'mse',
           'stdout': None,
           'stderr': None,
           'path_model': None,
           'file_M': None,
           'prop_weak': 1.0,
           'prop_clean': 1.0,
           }

dataset_functions = {'toy_example': load_toy_example,
                     'blobs': partial(load_weak_blobs, method='random_weak'),
                     'unbalanced': partial(load_weak_blobs,
                                           method='random_weak',
                                           n_samples=[100, 300, 1200, 1200,
                                                      5600, 1400]),
                     'blobs_webs': partial(load_weak_blobs,
                                           method='random_weak',
                                           n_samples=[1000, 3000, 12000, 12000,
                                                      56000, 14000],
                                           n_features=2099,
                                           n_classes=6,
                                           true_size=0.02),
                     'iris': partial(load_weak_iris, method='random_weak',
                                     true_size=0.3),
                     'webs': partial(load_webs,
                                     tfidf=True,
                                     standardize=True),
                     'classification': partial(load_classification,
                                               n_samples=10000,
                                               n_features=2,
                                               n_classes=2,
                                               n_informative=2,
                                               n_redundant=0,
                                               n_repeated=0,
                                               n_clusters_per_class=1),
                     }

def function_accepts_M(f):
    if type(dataset_functions[dataset]) is partial:
        return 'M' in inspect.getargspec(dataset_functions[dataset].func).args
    else:
        return 'M' in inspect.getargspec(dataset_functions[dataset]).args


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs a test with a toy
                                                example or a real dataset''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--architecture', dest='architecture', type=str,
                        default=DEFAULT['architecture'],
                        help='''Model architecture. Possible options are: lr
                        (logistic regression); or a MLP with the following
                        specification: mlp100m (Multilayer Perceptron
                        with 100 units in a hidden layer and Softmax), mlp60dm
                        (MLP with 60 hidden units, dropout 0.5 and SoftMax),
                        mlp30ds45dm, (MLP
                        with two hidden layers of 30 units, dropout of 0.5,
                        Sigmoid activation, layer of 45 units, dropout of 0.5,
                        and SoftMax).''')
    parser.add_argument('-c', '--epochs', dest='epochs', type=int,
                        default=DEFAULT['epochs'], help='Number of epochs')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        default=DEFAULT['dataset'],
                        help='''Name of the dataset to use: iris, toy_example,
                                blobs, unbalanced, webs, blobs_webs''')
    parser.add_argument('-e', '--stderr', dest='stderr',
                        default=DEFAULT['stderr'], action='store_true',
                        help='If the stderr needs to be redirected')
    parser.add_argument('-f', '--prop-weak', dest='prop_weak', type=float,
                        default=DEFAULT['prop_weak'],
                        help='Proportion of weak portion to keep')
    parser.add_argument('-g', '--prop-clean', dest='prop_clean', type=float,
                        default=DEFAULT['prop_clean'],
                        help='Proportion of clean portion to keep')
    parser.add_argument('-i', '--n-iterations', dest='n_iterations', type=int,
                        default=DEFAULT['n_iterations'],
                        help='Number of iterations to repeat the validation')
    parser.add_argument('-k', '--k-folds', dest='k_folds', type=int,
                        default=DEFAULT['k_folds'],
                        help='Number of folds for the cross-validation')
    parser.add_argument('-l', '--loss', dest='loss', type=str,
                        default=DEFAULT['loss'],
                        help='Number of iterations to repeat the validation')
    parser.add_argument('-M', '--file-M', dest='file_M', type=str,
                        default=DEFAULT['file_M'],
                        help='''File with a precomputed M''')
    parser.add_argument('-m', '--method', dest='method', type=str,
                        default=DEFAULT['method'],
                        help='''Learning method to use between,
                                Mproper, fully_supervised, fully_weak,
                                partially_weak, EM or OSL''')
    parser.add_argument('-o', '--stdout', dest='stdout',
                        default=DEFAULT['stdout'], action='store_true',
                        help='If the stdout needs to be redirected')
    parser.add_argument('-p', '--processes', dest='n_jobs', type=int,
                        default=DEFAULT['n_jobs'],
                        help='Number of concurrent processes')
    parser.add_argument('-r', '--path-results', dest='path_results', type=str,
                        default=DEFAULT['path_results'],
                        help='Path to the precomputed mixing matrix M')
    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=DEFAULT['seed'],
                        help='Seed for the random number generator')
    parser.add_argument('-t', '--path-model', dest='path_model', type=str,
                        default=DEFAULT['path_model'],
                        help='Path to the model and weights')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,
                        default=DEFAULT['verbose'],
                        help='Verbosity level being 0 the minimum value')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=DEFAULT['lr'],
                        help='Initial learning rate')
    parser.add_argument('--l1', dest='l1', type=float,
                        default=DEFAULT['l1'],
                        help='L1 regularization')
    parser.add_argument('--l2', dest='l2', type=float,
                        default=DEFAULT['l2'],
                        help='L2 regularization')
    parser.add_argument('--optimizer', dest='optimizer', type=str,
                        default=DEFAULT['optimizer'],
                        help=('Optimization method: rmsprop, adagrad, '
                              'adadelta, adam, adamax, nadam, tfoptimizer'))
    return parser.parse_args()


def main(dataset=DEFAULT['dataset'], seed=DEFAULT['seed'],
         verbose=DEFAULT['verbose'], method=DEFAULT['method'],
         path_results=DEFAULT['path_results'], n_jobs=DEFAULT['n_jobs'],
         n_iterations=DEFAULT['n_iterations'],
         k_folds=DEFAULT['k_folds'], architecture=DEFAULT['architecture'],
         loss=DEFAULT['loss'], stdout=DEFAULT['stdout'],
         stderr=DEFAULT['stderr'], epochs=DEFAULT['epochs'],
         path_model=DEFAULT['path_model'],
         file_M=DEFAULT['file_M'], prop_weak=DEFAULT['prop_weak'],
         prop_clean=DEFAULT['prop_clean'], lr=DEFAULT['lr'], l1=DEFAULT['l1'],
         l2=DEFAULT['l2'], optimizer=DEFAULT['optimizer']):

    diary = Diary(name=('{}_{}_{}'.format(dataset, method, architecture)),
                  path=path_results, overwrite=False, image_format='png',
                  fig_format='svg', stdout=stdout, stderr=stderr)

    print('Main arguments')
    print(locals())
    if dataset not in dataset_functions.keys():
        raise ValueError("Dataset not available: %s" % (dataset))

    if file_M is None:
        M = None
    else:
        M = np.loadtxt(file_M)

    if M is not None and function_accepts_M(dataset_functions[dataset]):
        training, validation, classes = dataset_functions[dataset](
                random_state=seed, M=M)
    else:
        training, validation, classes = dataset_functions[dataset](
                random_state=seed)

    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation

    if prop_weak < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, random_state=seed,
                                     train_size=prop_weak)
        one_member = [key for key, value in Counter(z_t).items() if value == 1]
        train_indx = np.where(z_t == one_member)
        X_t = np.concatenate((X_t, X_t[train_indx]))
        Z_t = np.concatenate((Z_t, Z_t[train_indx]))
        z_t = np.concatenate((z_t, z_t[train_indx]))
        train_indx, test_indx = sss.split(X_t, z_t).next()
        X_t, Z_t, z_t = X_t[train_indx], Z_t[train_indx], z_t[train_indx]
    if prop_clean < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, random_state=seed,
                                     train_size=prop_clean)
        one_member = [key for key, value in Counter(y_v).items() if value == 1]
        train_indx = np.where(z_t == one_member)
        X_v = np.concatenate((X_v, X_v[train_indx]))
        Z_v = np.concatenate((Z_v, Z_v[train_indx]))
        z_v = np.concatenate((z_v, z_v[train_indx]))
        Y_v = np.concatenate((Y_v, Y_v[train_indx]))
        y_v = np.concatenate((y_v, y_v[train_indx]))
        train_indx, test_indx = sss.split(X_v, y_v).next()
        X_v, Z_v, z_v, Y_v, y_v = X_v[train_indx], Z_v[train_indx], z_v[train_indx], Y_v[train_indx], y_v[train_indx]

    n_dataset = diary.add_notebook('dataset')
    n_dataset.add_entry(row=['dataset', dataset,
                       'n_samples_without_y', X_t.shape[0],
                       'n_samples_with_y', X_v.shape[0],
                       'n_features', X_t.shape[1],
                       'n_classes', Z_t.shape[1]])

    # TODO should I train with the same number of samples? or same number of
    # epochs?
    # Compute the equivalent number of epochs for fully_supervised
    # if method == 'fully_supervised':
    #     fully_s_set_size = (z_v.shape[0]/k_folds)*(k_folds-1)
    #     others_set_size = z_t.shape[0] + fully_s_set_size
    #     epochs = int((epochs*others_set_size)/fully_s_set_size)

    analyse_weak_labels(X_z=X_t, Z_z=Z_t, z_z=z_t, X_y=X_v, Z_y=Z_v,
                        z_y=z_v, Y_y=Y_v, y_y=y_v, random_state=seed,
                        verbose=verbose, classes=classes, method=method,
                        diary=diary, n_jobs=n_jobs, loss=loss,
                        n_iterations=n_iterations, k_folds=k_folds,
                        architecture=architecture, epochs=epochs,
                        path_model=path_model, file_M=file_M, lr=lr, l1=l1,
                        l2=l2, optimizer=optimizer)


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
