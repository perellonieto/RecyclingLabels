from experiments.data import load_toy_example
from experiments.data import load_weak_blobs
from experiments.data import load_weak_iris
from experiments.data import load_webs

from experiments.analysis import analyse_true_labels
from experiments.analysis import analyse_weak_labels

from experiments.diary import Diary

import argparse

import numpy as np

dataset_functions = {'toy_example': load_toy_example,
                     'blobs': load_weak_blobs,
                     'iris': load_weak_iris,
                     'webs': load_webs}


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs a test with a toy
                                                example or a real dataset''')
    parser.add_argument('-t', '--test', dest='test', type=str,
                        default='weak_labels',
                        help='''Test that needs to be run: true_labels or
                                weak_labels''')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        default='iris',
                        help='''Name of the dataset to use: iris, toy_example,
                                blobs, webs''')
    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=None,
                        help='Seed for the random number generator')
    parser.add_argument('-m', '--method', dest='method', type=str,
                        default='supervised',
                        help='''Learning method to use between supervised
                                Mproper or fully_supervised, fully_weak''')
    parser.add_argument('-M', '--path-M', dest='path_M', type=str,
                        default='data/M.npy',
                        help='Path to the precomputed mixing matrix M')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,
                        default=0,
                        help='Verbosity level being 0 the minimum value')
    parser.add_argument('-p', '--processes', dest='n_jobs', type=int,
                        default=None,
                        help='Number of concurrent processes')
    parser.add_argument('-i', '--iterations', dest='n_iterations', type=int,
                        default=2,
                        help='Number of iterations to repeat the validation')
    parser.add_argument('-k', '--k-folds', dest='k_folds', type=int,
                        default=2,
                        help='Number of folds for the cross-validation')
    return parser.parse_args()


def test_1a():
    training, validation = load_toy_example()
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    analyse_true_labels(X_v, Y_v, y_v, srandom_state0)


def test_1b():
    training, validation = load_weak_blobs()
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    analyse_true_labels(X_v, Y_v, y_v, random_state=0)


def test_1c():
    # Add class names as a return
    training, validation, classes = load_webs()
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    analyse_true_labels(X_v, Y_v, y_v, random_state=0, classes=classes)


def main(test, dataset, seed, verbose, method, path_M, n_jobs, n_iterations,
         k_folds):
    print('Main arguments')
    print(locals())
    if dataset not in dataset_functions.keys():
        raise ValueError("Dataset not available: %s" % (dataset))

    training, validation, classes = dataset_functions[dataset](random_state=seed)
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation

    diary = Diary(name=('{}_{}_{}'.format(test, dataset, method)),
                  path='results', overwrite=False, image_format='png',
                  fig_format='svg')

    entry_dataset = diary.add_notebook('dataset')
    entry_dataset(row=['n_samples_without_y', X_t.shape[0],
                       'n_samples_with_y', X_v.shape[0],
                       'n_features', X_t.shape[1],
                       'n_classes', Z_t.shape[1]])

    if test == 'true_labels':
        analyse_true_labels(X_v, Y_v, y_v, random_state=seed, verbose=verbose,
                            classes=classes, diary=diary, n_jobs=n_jobs,
                            n_iterations=n_iterations, k_folds=k_folds)
    elif test == 'weak_labels':
        analyse_weak_labels(X_z=X_t, Z_z=Z_t, z_z=z_t, X_y=X_v, Z_y=Z_v,
                            z_y=z_v, Y_y=Y_v, y_y=y_v, random_state=seed,
                            verbose=verbose, classes=classes, method=method,
                            diary=diary, n_jobs=n_jobs,
                            n_iterations=n_iterations, k_folds=k_folds)
    else:
        raise ValueError("Analysis not implemented: %s" % (test))


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
