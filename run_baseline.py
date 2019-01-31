import argparse
import inspect

import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from experiments.data import load_webs, load_weak_iris, load_weak_blobs, \
                             load_toy_example, load_classification, \
                             load_labelme, load_dataset_apply_model
from experiments.analysis import analyse_weak_labels, train_and_test_weak_labels
from experiments.diary import Diary


dataset_functions = {
    'toy_example': load_toy_example,
    'gaussians': partial(load_weak_blobs, method='random_weak',
                         n_samples=100000, n_features=2,
                         centers=[[-1, -1], [-1, 1], [1, 1]],
                         true_size=0.01, alpha=0.2, beta=0.3),
     'blobs': partial(load_weak_blobs, method='random_weak', n_samples=10000,
                      n_features=4, centers=6, true_size=0.01, alpha=0.2,
                      beta=0.3),
     'unbalanced': partial(load_weak_blobs, method='random_weak',
                           n_samples=[100, 300, 1200, 1200, 5600, 1400]),
     'blobs_webs': partial(load_weak_blobs, method='random_weak',
                           n_samples=[1000, 3000, 12000, 12000, 56000, 14000],
                           n_features=2099, centers=6, true_size=0.02),
     'iris': partial(load_weak_iris, method='random_weak', true_size=0.3),
     'webs': partial(load_webs, tfidf=True, categories=['parking', 'b2c',
                                                        'no_b2c', 'Other'],
                     standardize=True),
     'classification': partial(load_classification, n_samples=10000,
                               n_features=4, n_classes=6, n_informative=4,
                               n_redundant=0, n_repeated=0,
                               n_clusters_per_class=2, alpha=0.5, beta=0.3,
                               true_size=0.02),
     'labelme': partial(load_labelme, keep_valid_test=False),
     'iris_lr': partial(load_dataset_apply_model, dataset='iris'),
     'digits_lr': partial(load_dataset_apply_model, dataset='digits',
                          true_proportion=0.2),
     'cifar10_lr': partial(load_dataset_apply_model, dataset='cifar10',
                          true_proportion=0.2),
     'mnist_lr': partial(load_dataset_apply_model, dataset='mnist',
                          true_proportion=0.2),
     'fashion_mnist_lr': partial(load_dataset_apply_model, dataset='fashion_mnist',
                          true_proportion=0.2),
     }


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def float_or_none(x):
    if x == 'None':
        return None
    return float(x)

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
    parser.add_argument('-c', '--epochs', dest='epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('-d', '--dataset', dest='dataset', type=str,
                        default='iris',
                        help='''Name of the dataset to use: iris, toy_example,
                                blobs, unbalanced, webs, blobs_webs''')
    parser.add_argument('-e', '--stderr', dest='stderr',
                        default=None, action='store_true',
                        help='If the stderr needs to be redirected')
    parser.add_argument('-f', '--prop-weak', dest='prop_weak', type=float,
                        default=1.0,
                        help='Proportion of weak portion to keep')
    parser.add_argument('-g', '--prop-clean', dest='prop_clean', type=float,
                        default=1.0,
                        help='Proportion of clean portion to keep')
    parser.add_argument('--prop-test', dest='prop_test', type=float,
                        default=0.2,
                        help='''Proportion of test data from portion of data
                        with true labels''')
    parser.add_argument('-i', '--n-iterations', dest='n_iterations', type=int,
                        default=2,
                        help='Number of iterations to repeat the validation')
    parser.add_argument('-k', '--k-folds', dest='k_folds', type=int,
                        default=2,
                        help='Number of folds for the cross-validation')
    parser.add_argument('-l', '--loss', dest='loss', type=str,
                        default='log_loss',
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
                        default=None, action='store_true',
                        help='If the stdout needs to be redirected')
    parser.add_argument('-p', '--processes', dest='n_jobs', type=int,
                        default=None,
                        help='Number of concurrent processes')
    parser.add_argument('-r', '--path-results', dest='path_results', type=str,
                        default='results',
                        help='Path to the precomputed mixing matrix M')
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=42,
                        help='Seed for the random number generator')
    parser.add_argument('-t', '--path-model', dest='path_model', type=str,
                        default=None,
                        help='Path to the model and weights')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int, default=0,
                        help='Verbosity level being 0 the minimum value')
    parser.add_argument('--lr', dest='lr', type=float, default=1.0,
                        help='Initial learning rate')
    parser.add_argument('--l1', dest='l1', type=float, default=0.0,
                        help='L1 regularization')
    parser.add_argument('--l2', dest='l2', type=float, default=0.001,
                        help='L2 regularization')
    parser.add_argument('--optimizer', dest='optimizer', type=str,
                        default='adam',
                        help=('Optimization method: rmsprop, adagrad, '
                              'adadelta, adam, adamax, nadam, tfoptimizer'))
    parser.add_argument('--decay', dest='decay', type=float, default=0.5,
                        help='decay')
    parser.add_argument('--momentum', dest='momentum', type=float,
                        default=0.5,
                        help='Momentum')
    parser.add_argument('--rho', dest='rho', type=float, default=0.9,
                        help='rho')
    parser.add_argument('--epsilon', dest='epsilon', type=float_or_none,
                        default=None,
                        help='epsilon')
    parser.add_argument('--nesterov', dest='nesterov', type=str2bool,
                        default=True,
                        help='nesterov')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=100,
                        help='batch_size')
    return parser.parse_args()


def main(dataset, seed, verbose, method, path_results, n_jobs, n_iterations,
         k_folds, architecture, loss, stdout, stderr, epochs, path_model,
         file_M, prop_weak, prop_clean, prop_test, lr, l1, l2, optimizer,
         momentum, decay, nesterov, batch_size, rho, epsilon):

    # Store the original main arguments
    main_arguments = locals()

    diary = Diary(name=('{}_{}_{}'.format(dataset, method, architecture)),
                  path=path_results, overwrite=False, image_format='png',
                  fig_format='svg', stdout=stdout, stderr=stderr)

    print('Main arguments')
    print(main_arguments)
    if dataset not in dataset_functions.keys():
        raise ValueError("Dataset not available: %s" % (dataset))

    n_arg = diary.add_notebook('arguments')
    n_arg.add_entry(row=main_arguments)

    if file_M is None:
        M = None
    else:
        M = np.loadtxt(file_M)

    if M is not None and function_accepts_M(dataset_functions[dataset]):
        training, validation, test, classes = dataset_functions[dataset](
                random_state=seed, M=M)
    else:
        training, validation, test, classes = dataset_functions[dataset](
                random_state=seed)

    X_t, Z_t, z_t = training
    X_v, Z_v, z_v, Y_v, y_v = validation

    # Get test partition
    if test is None:
        sss = StratifiedShuffleSplit(n_splits=1, random_state=seed,
                                     train_size=(1. - prop_test),
                                     test_size=prop_test)
        val_indx, test_indx = next(sss.split(X_v, y_v))
        print('True labels: Validation original partition size = {}'.format(len(val_indx)))
        print('True labels: Test original partition size = {}'.format(len(test_indx)))
        # test partition
        X_te, Z_te, z_te = X_v[test_indx], Z_v[test_indx], z_v[test_indx]
        Y_te, y_te = Y_v[test_indx], y_v[test_indx]
        # Validation partition
        X_v, Z_v, z_v = X_v[val_indx], Z_v[val_indx], z_v[val_indx]
        Y_v, y_v = Y_v[val_indx], y_v[val_indx]
    else:
        X_te, Y_te, y_te = test


    if prop_weak < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, random_state=seed,
                                     train_size=prop_weak,
                                     test_size=(1. - prop_weak))
        one_member = [key for key, value in Counter(z_t).items() if value == 1]
        train_indx = np.where(z_t == one_member)
        X_t = np.concatenate((X_t, X_t[train_indx]))
        Z_t = np.concatenate((Z_t, Z_t[train_indx]))
        z_t = np.concatenate((z_t, z_t[train_indx]))
        train_indx, test_indx = next(sss.split(X_t, z_t))
        X_t, Z_t, z_t = X_t[train_indx], Z_t[train_indx], z_t[train_indx]

    if prop_clean < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, random_state=seed,
                                     train_size=prop_clean,
                                     test_size=(1. - prop_clean))
        one_member = [key for key, value in Counter(y_v).items() if value == 1]
        train_indx = np.where(z_t == one_member)
        X_v = np.concatenate((X_v, X_v[train_indx]))
        Z_v = np.concatenate((Z_v, Z_v[train_indx]))
        z_v = np.concatenate((z_v, z_v[train_indx]))
        Y_v = np.concatenate((Y_v, Y_v[train_indx]))
        y_v = np.concatenate((y_v, y_v[train_indx]))
        train_indx, test_indx = next(sss.split(X_v, y_v))
        X_v, Z_v, z_v, Y_v, y_v = X_v[train_indx], Z_v[train_indx], z_v[train_indx], Y_v[train_indx], y_v[train_indx]
    print('True labels: Validation partition size = {}'.format(len(y_v)))
    print('True labels: Test partition size = {}'.format(len(y_te)))

    # There is a problem with the validation and test size
    n_dataset = diary.add_notebook('dataset')
    n_dataset.add_entry(row=['dataset', dataset,
                       'n_samples_without_y', X_t.shape[0],
                       'n_samples_with_y', X_v.shape[0] + X_te.shape[0],
                       'n_features', X_t.shape[1],
                       'n_classes', Z_t.shape[1],
                       'train_size', X_t.shape[0],
                       'valid_size', X_v.shape[0],
                       'test_size', X_te.shape[0]])

    # TODO should I train with the same number of samples? or same number of
    # epochs?
    # Compute the equivalent number of epochs for fully_supervised
    # if method == 'fully_supervised':
    #     fully_s_set_size = (z_v.shape[0]/k_folds)*(k_folds-1)
    #     others_set_size = z_t.shape[0] + fully_s_set_size
    #     epochs = int((epochs*others_set_size)/fully_s_set_size)

    best_epoch = analyse_weak_labels(X_z=X_t, Z_z=Z_t, z_z=z_t, X_y=X_v, Z_y=Z_v,
                        z_y=z_v, Y_y=Y_v, y_y=y_v, random_state=seed,
                        verbose=verbose, classes=classes, method=method,
                        diary=diary, n_jobs=n_jobs, loss=loss,
                        n_iterations=n_iterations, k_folds=k_folds,
                        architecture=architecture, epochs=epochs,
                        path_model=path_model, file_M=file_M, lr=lr, l1=l1,
                        l2=l2, optimizer=optimizer, momentum=momentum,
                        decay=decay, nesterov=nesterov, batch_size=batch_size,
                        rho=rho, epsilon=epsilon)

    if best_epoch == 0:
        best_epoch = 1

    train_and_test_weak_labels(X_z=X_t, Z_z=Z_t, z_z=z_t, X_y=X_v, Z_y=Z_v,
                               z_y=z_v, Y_y=Y_v, y_y=y_v, X_te=X_te,
                               Y_te=Y_te, y_te=y_te,
                               random_state=seed, verbose=verbose,
                               classes=classes, method=method, diary=diary,
                               loss=loss, architecture=architecture,
                               n_iterations=n_iterations,
                               epochs=best_epoch, path_model=path_model,
                               file_M=file_M, lr=lr, l1=l1, l2=l2,
                               optimizer=optimizer, momentum=momentum,
                               decay=decay, nesterov=nesterov,
                               batch_size=batch_size, rho=rho, epsilon=epsilon)


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
