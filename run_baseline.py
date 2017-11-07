import argparse

import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from experiments.data import load_webs, load_weak_iris, load_weak_blobs, \
                             load_toy_example
from experiments.analysis import analyse_weak_labels
from experiments.diary import Diary

dataset_functions = {'toy_example': load_toy_example,
                     'blobs': partial(load_weak_blobs, method='random_weak'),
                     'unbalanced': partial(load_weak_blobs,
                                           method='random_weak',
                                           n_samples=[100, 300, 1200, 1200, 5600, 1400]),
                     'blobs_webs': partial(load_weak_blobs,
                                           method='random_weak',
                                           n_samples=[1000, 3000, 12000, 12000,
                                                      56000, 14000],
                                           n_features=2099,
                                           n_classes=6,
                                           true_size=0.02),
                     'iris': partial(load_weak_iris, method='random_weak',
                         true_size=0.3),
                     'webs': load_webs}


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs a test with a toy
                                                example or a real dataset''')
    parser.add_argument('-a', '--architecture', dest='architecture', type=str,
                        default='lr',
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
                        default=200, help='Number of epochs')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        default='iris',
                        help='''Name of the dataset to use: iris, toy_example,
                                blobs, unbalanced, webs, blobs_webs''')
    parser.add_argument('-e', '--stderr', dest='stderr',
                        default=False, action='store_true',
                        help='If the stderr needs to be redirected')
    parser.add_argument('-f', '--prop-weak', dest='prop_weak', type=float,
                        default=1.0,
                        help='Proportion of weak portion to keep')
    parser.add_argument('-g', '--prop-clean', dest='prop_clean', type=float,
                        default=1.0,
                        help='Proportion of clean portion to keep')
    parser.add_argument('-i', '--iterations', dest='n_iterations', type=int,
                        default=2,
                        help='Number of iterations to repeat the validation')
    parser.add_argument('-k', '--k-folds', dest='k_folds', type=int,
                        default=2,
                        help='Number of folds for the cross-validation')
    parser.add_argument('-l', '--loss', dest='loss', type=str,
                        default='mse',
                        help='Number of iterations to repeat the validation')
    parser.add_argument('-M', '--file-M', dest='file_M', type=str,
                        default=None,
                        help='''File with a precomputed M''')
    parser.add_argument('-m', '--method', dest='method', type=str,
                        default='fully_supervised',
                        help='''Learning method to use between,
                                Mproper, fully_supervised, fully_weak,
                                partially_weak, EM or OSL''')
    parser.add_argument('-o', '--stdout', dest='stdout',
                        default=False, action='store_true',
                        help='If the stdout needs to be redirected')
    parser.add_argument('-p', '--processes', dest='n_jobs', type=int,
                        default=None,
                        help='Number of concurrent processes')
    parser.add_argument('-r', '--path-results', dest='path', type=str,
                        default='results',
                        help='Path to the precomputed mixing matrix M')
    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=None,
                        help='Seed for the random number generator')
    parser.add_argument('-t', '--path-model', dest='path_model', type=str,
                        default=None,
                        help='Path to the model and weights')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,
                        default=0,
                        help='Verbosity level being 0 the minimum value')
    return parser.parse_args()


def main(dataset, seed, verbose, method, path, n_jobs, n_iterations,
         k_folds, architecture, loss, stdout, stderr, epochs, path_model,
         file_M, prop_weak, prop_clean):

    diary = Diary(name=('{}_{}_{}'.format(dataset, method, architecture)),
                  path=path, overwrite=False, image_format='png',
                  fig_format='svg', stdout=stdout, stderr=stderr)

    print('Main arguments')
    print(locals())
    if dataset not in dataset_functions.keys():
        raise ValueError("Dataset not available: %s" % (dataset))

    training, validation, classes = dataset_functions[dataset](random_state=seed)
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
    n_dataset.add_entry(row=['name', dataset,
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
                        path_model=path_model, file_M=file_M)


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
