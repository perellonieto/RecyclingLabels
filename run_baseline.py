from experiments.data import load_toy_example
from experiments.data import load_blobs
from experiments.data import load_webs

from experiments.analysis import analyse_true_labels

import argparse

analysis_functions = {'true_labels': analyse_true_labels}
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
    analyse_true_labels(load_toy_example, seed=0)


def test_1b():
    analyse_true_labels(load_blobs, seed=0)


def test_1c():
    analyse_true_labels(load_webs, seed=0)


def main(test, dataset, seed, verbose):
    print('Main arguments')
    print(locals())
    analysis_functions[test](dataset_functions[dataset], seed, verbose)


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
