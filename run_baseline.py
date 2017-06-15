from experiments.data import load_toy_example
from experiments.data import load_blobs
from experiments.data import load_webs

from experiments.analysis import analyse_true_labels
from experiments.analysis import analyse_weak_labels

from experiments.diary import Diary

import argparse

import numpy as np

dataset_functions = {'toy_example': load_toy_example,
                     'blobs': load_blobs,
                     'webs': load_webs}


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs a test with a toy
                                                example or a real dataset''')
    parser.add_argument('-t', '--test', dest='test', type=str,
                        default='true_labels',
                        help='Test that needs to be run')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        default='toy_example',
                        help='Name of the dataset to use')
    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=None,
                        help='Seed for the random number generator')
    parser.add_argument('-m', '--method', dest='method', type=str,
                        default='supervised',
                        help=('Learning method to use between supervised'
                              'Mproper or quasi_IPL'))
    parser.add_argument('-M', '--path-M', dest='path_M', type=str,
                        default='data/M.npy',
                        help='Path to the precomputed mixing matrix M')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,
                        default=0,
                        help='Verbosity level being 0 the minimum value')
    return parser.parse_args()


def test_1a():
    training, validation = load_toy_example()
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    analyse_true_labels(X_v, Y_v, y_v, seed=0)


def test_1b():
    training, validation = load_blobs()
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    analyse_true_labels(X_v, Y_v, y_v, seed=0)


def test_1c():
    # Add class names as a return
    training, validation, classes = load_webs()
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    analyse_true_labels(X_v, Y_v, y_v, seed=0, classes=classes)


def main(test, dataset, seed, verbose, method, path_M):
    print('Main arguments')
    print(locals())
    if dataset not in dataset_functions.keys():
        raise ValueError("Dataset not available: %s" % (dataset))

    training, validation, classes = dataset_functions[dataset](seed=seed)
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation

    diary = Diary(name=test, path='results', overwrite=False,
                  image_format='png', fig_format='svg')

    entry_dataset = diary.add_notebook('dataset')
    entry_dataset(row=['n_samples_train', X_t.shape[0],
                       'n_samples_val', X_v.shape[0],
                       'n_features', X_t.shape[1], 'n_classes', Z_t.shape[1]])

    if test == 'true_labels':
        analyse_true_labels(X_v, Y_v, y_v, seed=seed, verbose=verbose,
                            classes=classes, diary=diary)
    elif test == 'weak_labels':
        M = np.load(path_M)
        M = M.item()
        analyse_weak_labels(X_train=X_t, Z_train=Z_t, z_train=z_t, X_val=X_v,
                            Z_val=Z_v, z_val=z_v, Y_val=Y_v, y_val=y_v,
                            seed=seed, verbose=verbose, classes=classes,
                            method=method, M=M, diary=diary)
    else:
        raise ValueError("Analysis not implemented: %s" % (test))


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
