from experiments.data import load_toy_example
from experiments.data import load_blobs
from experiments.data import load_webs

from experiments.analysis import analyse_true_labels

import argparse

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
                        help='Test that needs to be run')
    parser.add_argument('-s', '--seed', dest='seed', type=int,
                        default=None,
                        help='Test that needs to be run')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,
                        default=0,
                        help='Verbosity level with 0 the minimum')
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
    training, validation = load_webs()
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    analyse_true_labels(X_v, Y_v, y_v, seed=0)


def main(test, dataset, seed, verbose):
    print('Main arguments')
    print(locals())
    if dataset not in dataset_functions.keys():
        raise ValueError("Dataset not available: %s" % (dataset))

    training, validation = dataset_functions[dataset](seed=seed)
    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation
    if test == 'true_labels':
        analyse_true_labels(X_v, Y_v, y_v, seed=seed, verbose=verbose)
    else:
        raise ValueError("Analysis not implemented: %s" % (test))


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
