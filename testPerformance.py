#!/usr/bin/python

import pprint

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam

from keras.utils.vis_utils import model_to_dot

from tests import test_1

from testUtils import binarizeWeakLabels
from testUtils import create_model

from testData import load_toy_example
from testData import load_blobs
from testData import load_webs

seed = 42


def main():
    X_train, Z_train, z_train, X_val, Z_val, z_val, Y_val = load_webs()

    n_classes = Y_val.shape[1]

    # Training model
    model = KerasClassifier(build_fn=create_model, verbose=0)

    param_grids = [{'input_dim': [X_train.shape[1]],
                    'output_size': [n_classes],
                    'optimizer': ['sgd'],
                    'init': ['glorot_uniform'],
                    'lr': [1.0, 0.1, 0.01, 0.001],
                    'momentum': [0.0, 0.1],
                    'decay': [0.0, 0.1],
                    'nesterov': [True],
                    'epochs': [20],
                    'batch_size': [10]
                    }]

    grid = GridSearchCV(model, param_grids, verbose=2, n_jobs=-1)
    grid_search_results = grid.fit(X_val, Y_val)

    pp = pprint.PrettyPrinter(indent=4)
    print("Best score = {}".format(grid_search_results.best_score_))
    print("Best parameters:")
    print(pp.pprint(grid_search_results.best_params_))


if __name__ == '__main__':
    test_1b()
    #main()
