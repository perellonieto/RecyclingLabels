import os
import pprint
import inspect
import multiprocessing

import numpy as np
from collections import Counter

from sklearn.model_selection import StratifiedKFold
# TODO Change to model_selection
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from experiments.models import create_model, MyKerasClassifier
from experiments.visualizations import plot_confusion_matrix, \
                                       plot_multilabel_scatter, \
                                       plot_errorbar
from experiments.diary import Diary
from experiments.diary import SharedDiary
from experiments.utils import merge_dicts
from experiments.metrics import compute_expected_error, compute_error_matrix

from wlc.WLweakener import computeVirtual, computeM, estimate_M, weak_to_index


def save_model(path, model, process_id):
    print("Saving model to {}".format(os.path.join(path, "model_{}.h5".format(process_id))))
    model_json = model.to_json()
    with open(os.path.join(path, "model_{}.json".format(process_id)), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(path, "model_{}.h5".format(process_id)))


# TODO take a look that everything is ok
def train_weak_Mproper_test_results(parameters):
    """Train a model using the Mproper approach:

    1. Learn a mixing matrix using training with weak and true labels
        - M = f(Z_y_t, Y_y_t)
    2. Compute virtual labels for training set only with weak labels
        - V_z_t = f(M, Z_z_t)
    3. Train a model using the training set with virtual and true labels
        - model.fit([X_z_t and X_y_t], [V_z_t and Y_y_t])
    4. Evaluate the model in the validation set with true labels
        - y_pred = model.predict(X_y_v)
        - evaluate(y_pred, Y_y_v)

    Parameters
    ----------
    X_z_t : array-like, with shape (n_training_samples_without_y, n_dim)
        Matrix with features used for training with only weak labels available

    Z_z_t : array-like, with shape (n_training_samples_without_y, n_classes)
        Weak labels for training

    X_y_t : array-like, with shape (n_training_samples_with_y, n_dim)
        Matrix with features used for training with weak and true labels

    Z_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        Weak labels for training with the true labels available

    Y_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        True labels for training

    X_y_v : array-like, with shape (n_validation_samples_with_y, n_dim)
        Matrix with features used for validation with weak and true labels

    Y_y_v : array-like, with shape (n_validation_samples_with_y, n_classes)
        True labels for validation

    n_extra: Notebook
        Notebook to save any extra information

    M: mixing matrix M
    """
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments, s_diary_vars, M = parameters

    s_diary = SharedDiary(s_diary_vars, unique_id=process_id)
    n_extra = s_diary.add_notebook('extra')

    n_c = Y_y_v.shape[1]
    categories = range(n_c)

    verbose = fit_arguments.get('verbose', 0)

    # 1. Learn a mixing matrix using training with weak and true labels
    if M is None:
        #M = estimate_M(Z_y_t, Y_y_t, categories, reg='Complete')
        M = estimate_M(Z_y_t, Y_y_t, categories, reg='Partial')
    n_extra.add_entry(row={'M': "\n{}".format(np.round(M, decimals=3))})
    # 2. Compute virtual labels for training set only with weak labels
    V_z_t = computeVirtual(Z_z_t, c=n_c, method='Mproper', M=M)
    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 3. Train a model using the training set with virtual and true labels
    V_t = np.concatenate((V_z_t, Y_y_t), axis=0)
    X_t = np.concatenate((X_z_t, X_y_t), axis=0)
    np.random.seed(process_id)
    X_t, V_t = shuffle(X_t, V_t)
    # Add validation results during training
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_t, V_t, X_y_t=X_y_t, Y_y_t=Y_y_t, **fit_arguments)
    # 4. Evaluate the model in the validation set with true labels
    # FIXME this outputs classes from 0 to #classes - 1
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('MP: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history.history}

    save_model(s_diary.path, classifier.model, process_id)

    return results


# TODO take a look that everything is ok
def train_weak_EM_test_results(parameters):
    """Train a model using the Expectation Maximization approach:

        1. Learn a mixing matrix using training with weak and true labels
        2. Compute the index of each sample relating it to the corresponding
        row of the new mixing matrix
            - Needs to compute the individual M and their weight q
        3. Give the mixing matrix to the model for future use
        4. Train model using all the sets with instead of labels the index of
        the corresponding rows of the mixing matrix
        5. Evaluate the model in the validation set with true labels

    Parameters
    ----------
    X_z_t : array-like, with shape (n_training_samples_without_y, n_dim)
        Matrix with features used for training with only weak labels available

    Z_z_t : array-like, with shape (n_training_samples_without_y, n_classes)
        Weak labels for training

    X_y_t : array-like, with shape (n_training_samples_with_y, n_dim)
        Matrix with features used for training with weak and true labels

    Z_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        Weak labels for training with the true labels available

    Y_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        True labels for training

    X_y_v : array-like, with shape (n_validation_samples_with_y, n_dim)
        Matrix with features used for validation with weak and true labels

    Y_y_v : array-like, with shape (n_validation_samples_with_y, n_classes)
        True labels for validation

    n_extra: Notebook
        Notebook to save any extra information

    M: mixing matrix M
    """
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments, s_diary_vars, M = parameters

    s_diary = SharedDiary(s_diary_vars, unique_id=process_id)
    n_extra = s_diary.add_notebook('extra')

    n_c = Y_y_v.shape[1]
    categories = range(n_c)

    verbose = fit_arguments.get('verbose', 0)

    # 1. Learn a mixing matrix using training with weak and true labels
    if M is None:
        # M_0 = estimate_M(Z_y_t, Y_y_t, categories, reg='Complete')
        M_0 = estimate_M(Z_y_t, Y_y_t, categories, reg='Partial')
        # M_0 = estimate_M(Z_y_t, Y_y_t, categories, reg=None)
    else:
        M_0 = M
    M_1 = computeM(c=n_c, method='supervised')
    q_0 = X_z_t.shape[0] / float(X_z_t.shape[0] + X_y_t.shape[0])
    q_1 = X_y_t.shape[0] / float(X_z_t.shape[0] + X_y_t.shape[0])
    M = np.concatenate((q_0*M_0, q_1*M_1), axis=0)
    #  2. Compute the index of each sample relating it to the corresponding
    #     row of the new mixing matrix
    #      - Needs to compute the individual M and their weight q
    Z_z_t_index = weak_to_index(Z_z_t, method='Mproper')
    Y_y_t_index = weak_to_index(Y_y_t, method='supervised')

    n_extra.add_entry(row={'q0': q_0, 'q1': q_1})
    n_extra.add_entry(row={'M_0': "\n{}".format(np.round(M_0, decimals=3))})
    n_extra.add_entry(row={'M_1': "\n{}".format(np.round(M_1, decimals=3))})
    n_extra.add_entry(row={'M': "\n{}".format(np.round(M, decimals=3))})
    n_extra.add_entry(row={'Z_y_t': "\n{}".format(np.round(Z_y_t[:5]))})
    n_extra.add_entry(row={'Z_z_t_index': Z_z_t_index[:5]})
    n_extra.add_entry(row={'Z_z_t': "\n{}".format(np.round(Z_z_t[:5]))})
    n_extra.add_entry(row={'Y_y_t_index': Y_y_t_index[:5]})
    n_extra.add_entry(row={'Y_y_t': "\n{}".format(np.round(Y_y_t[:5]))})
    # TODO was the next line to debug?
    #Z_y_t_index = weak_to_index(Z_y_t, method='Mproper')

    # 3. Give the mixing matrix to the model for future use
    #    I need to give the matrix M to the fit function
    # 4. Train model using all the sets with instead of labels the index of
    #    the corresponding rows of the mixing matrix
    Z_index_t = np.concatenate((Z_z_t_index,
                                Y_y_t_index + M_0.shape[0]))
    np.random.seed(process_id)
    X_t = np.concatenate((X_z_t, X_y_t), axis=0)
    X_t, Z_index_t = shuffle(X_t, Z_index_t)
    # Add validation results during training
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_t, Z_index_t, M=M, X_y_t=X_y_t, Y_y_t=Y_y_t, **fit_arguments)
    # 5. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history.history}

    save_model(s_diary.path, classifier.model, process_id)

    return results


# TODO take a look that everything is ok
def train_weak_fully_supervised_test_results(parameters):
    """Train a model using the fully supervised approach:

        1. Train model with the training set that has true labels
            - model.fit(X_y_t, Y_y_t)
        2. Evaluate the model in the validation set with true labels
            - y_pred = model.predict(X_y_v)
            - evaluate(y_pred, Y_y_v)

    Parameters
    ----------
    X_z_t : array-like, with shape (n_training_samples_without_y, n_dim)
        Matrix with features used for training with only weak labels available

    Z_z_t : array-like, with shape (n_training_samples_without_y, n_classes)
        Weak labels for training

    X_y_t : array-like, with shape (n_training_samples_with_y, n_dim)
        Matrix with features used for training with weak and true labels

    Z_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        Weak labels for training with the true labels available

    Y_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        True labels for training

    X_y_v : array-like, with shape (n_validation_samples_with_y, n_dim)
        Matrix with features used for validation with weak and true labels

    Y_y_v : array-like, with shape (n_validation_samples_with_y, n_classes)
        True labels for validation

    n_extra: Notebook
        Notebook to save any extra information
    """
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments, s_diary_vars = parameters

    s_diary = SharedDiary(s_diary_vars, unique_id=process_id)

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has true labels
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_y_t, Y_y_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('FS: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history.history}

    save_model(s_diary.path, classifier.model, process_id)

    return results


def train_weak_fully_weak_test_results(parameters):
    """Train a model using the fully supervised approach:

        1. Train model with the training set that has weak labels
            - model.fit([X_z_t and X_y_t], [Z_z_t and Z_y_t])
        2. Evaluate the model in the validation set with true labels
            - y_pred = model.predict(X_y_v)
            - evaluate(y_pred, Y_y_v)

    Parameters
    ----------
    X_z_t : array-like, with shape (n_training_samples_without_y, n_dim)
        Matrix with features used for training with only weak labels available

    Z_z_t : array-like, with shape (n_training_samples_without_y, n_classes)
        Weak labels for training

    X_y_t : array-like, with shape (n_training_samples_with_y, n_dim)
        Matrix with features used for training with weak and true labels

    Z_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        Weak labels for training with the true labels available

    Y_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        True labels for training

    X_y_v : array-like, with shape (n_validation_samples_with_y, n_dim)
        Matrix with features used for validation with weak and true labels

    Y_y_v : array-like, with shape (n_validation_samples_with_y, n_classes)
        True labels for validation

    n_extra: Notebook
        Notebook to save any extra information
    """
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments, s_diary_vars = parameters

    s_diary = SharedDiary(s_diary_vars, unique_id=process_id)

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has weak labels
    X_z_t = np.concatenate([X_z_t, X_y_t])
    Z_z_t = np.concatenate([Z_z_t, Z_y_t])
    X_z_t, Z_z_t = shuffle(X_z_t, Z_z_t, random_state=process_id)
    # Add validation results during training
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_z_t, Z_z_t, X_y_t=X_y_t, Y_y_t=Y_y_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('FW: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history.history}

    save_model(s_diary.path, classifier.model, process_id)

    return results


def train_weak_partially_weak_test_results(parameters):
    """Train a model using the fully supervised approach:

        1. Train model with the training set with weak labels and true labels
            when available
            - model.fit([X_z_t and X_y_t], [Z_z_t and Y_y_t])
        2. Evaluate the model in the validation set with true labels
            - y_pred = model.predict(X_y_v)
            - evaluate(y_pred, Y_y_v)

    Parameters
    ----------
    X_z_t : array-like, with shape (n_training_samples_without_y, n_dim)
        Matrix with features used for training with only weak labels available

    Z_z_t : array-like, with shape (n_training_samples_without_y, n_classes)
        Weak labels for training

    X_y_t : array-like, with shape (n_training_samples_with_y, n_dim)
        Matrix with features used for training with weak and true labels

    Z_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        Weak labels for training with the true labels available

    Y_y_t : array-like, with shape (n_training_samples_with_y, n_classes)
        True labels for training

    X_y_v : array-like, with shape (n_validation_samples_with_y, n_dim)
        Matrix with features used for validation with weak and true labels

    Y_y_v : array-like, with shape (n_validation_samples_with_y, n_classes)
        True labels for validation

    n_extra: Notebook
        Notebook to save any extra information
    """
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments, s_diary_vars = parameters

    s_diary = SharedDiary(s_diary_vars, unique_id=process_id)

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has weak labels with the true
    #    labels when available
    X_z_t = np.concatenate([X_z_t, X_y_t])
    Z_z_t = np.concatenate([Z_z_t, Y_y_t])
    X_z_t, Z_z_t = shuffle(X_z_t, Z_z_t, random_state=process_id)
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_z_t, Z_z_t, X_y_t=X_y_t, Y_y_t=Y_y_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('PW: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history.history}

    save_model(s_diary.path, classifier.model, process_id)

    return results


# TODO add other methods
def analyse_weak_labels(X_z, Z_z, z_z, X_y, Z_y, z_y, Y_y, y_y, classes,
                        n_iterations=2, k_folds=2, diary=None, verbose=0,
                        random_state=None, method='Mproper', n_jobs=None,
                        architecture='lr', loss='mse', epochs=200,
                        path_model=None, file_M=None, lr=1.0, l1=0.0,
                        l2=0.001, optimizer='rmsprop'):
    """ Trains a Feed-fordward neural network using cross-validation

    The training is done with the weak labels on the training set and
    the model is evaluated with the true labels in the validation set

    Parameters
    ----------
    X_z : array-like, with shape (n_samples_without_y, n_dim)
        Matrix with features with only weak labels available

    Z_z : array-like, with shape (n_samples_without_y, n_classes)
        Weak labels in a binary matrix

    z_z : array-like, with shape (n_samples_without_y, )
        Weak labels in decimal

    X_y : array-like, with shape (n_samples_with_y, n_dim)
        Matrix with features with both weak and true labels available

    Z_y : array-like, with shape (n_samples_with_y, n_classes)
        Weak labels in a binary matrix

    z_y : array-like, with shape (n_samples_with_y, )
        Weak labels in decimal

    Y_y : array-like, with shape (n_samples_with_y, n_classes)
        True labels in a binary matrix

    y_y : array-like, with shape (n_samples_with_y, )
        True labels in decimal

    path_model : String or None
        If a string, it is the path with the folder that contains a model.json
        and the weights.h5 that will be used as initialisation for the
        training.
    """
    # Test performance on validation true labels
    # ## Create a Diary for all the logs and results
    if diary is None:
        diary = Diary(name='weak_labels', path='results', overwrite=False,
                      image_format='png', fig_format='svg')

    n_model = diary.add_notebook('model')
    n_val = diary.add_notebook('validation')
    n_tra = diary.add_notebook('training')
    n_extra = diary.add_notebook('extra')

    s_diary_vars = diary.get_shared_vars()

    n_f = X_z.shape[1]
    n_c = Y_y.shape[1]


    prior_y = np.true_divide(Y_y.sum(axis=0), Y_y.sum())
    n_extra.add_entry(row={'prior_distribution': prior_y})
    brier_score = lambda yp, yt: (yp - yt)**2
    error_matrix = compute_error_matrix(prior_y, brier_score)
    expected_bs = compute_expected_error(prior_y, error_matrix)
    n_extra.add_entry(row={'expected_brier_score': expected_bs})
    log_loss = lambda yp, yt: -yt*np.log(yp)
    error_matrix = compute_error_matrix(prior_y, log_loss)
    expected_ll = compute_expected_error(prior_y, error_matrix)
    n_extra.add_entry(row={'expected_log_loss': expected_ll})
    n_extra.add_entry(row={'expected_accuracy': max(prior_y)})
    n_extra.add_entry(row={'mean_acc_Y_Z': (Z_y == Y_y).mean(axis=0)})

    # If dimension is 2, we draw a 2D scatterplot
    if n_f >= 2:
        n_subsample = 50 if X_y.shape[0] > 50 else X_y.shape[0]
        fig = plot_multilabel_scatter(X_y[:n_subsample],
                                      Y_y[:n_subsample], title='True labels')
        diary.save_figure(fig, filename='true_labels')

        fig = plot_multilabel_scatter(X_y[:n_subsample],
                                      Z_y[:n_subsample], title='Weak labels')
        diary.save_figure(fig, filename='weak_labels')

    if method == 'OSL':
        training_method = 'OSL'
    elif method in ['Mproper','fully_weak', 'partially_weak']:
        training_method = 'weak'
    elif method == 'EM':
        training_method = 'EM'
    elif method == 'fully_supervised':
        training_method = 'supervised'
    else:
        raise(ValueError('Method unknown {}'.format(method)))

    LOSS_MAP = dict(mse='mean_squared_error', wbs='w_brier_score',
                    bs='brier_score')

    loss = LOSS_MAP[loss]

    # Parameters for the multiprocessing training and validation
    params = {'input_dim': n_f,
              'output_size': n_c,
              'optimizer': optimizer,
              'loss': loss,
              'init': 'glorot_uniform',
              'lr': lr,
              'l1': l1,
              'l2': l2,
              'momentum': 0.5,
              'decay': 0.5,
              'nesterov': True,
              'epochs': epochs,
              'batch_size': 100,
              'verbose': verbose,
              'random_state': random_state,
              'training_method': training_method,
              'architecture': architecture,
              'path_model': path_model
              }
    #params = {'input_dim': n_f,
    #          'output_size': n_c,
    #          'optimizer': 'sgd',
    #          'loss': loss,
    #          'init': 'glorot_uniform',
    #          'lr': 1.0,
    #          'momentum': 0,
    #          'decay': 0,
    #          'nesterov': False,
    #          'epochs': epochs,
    #          'batch_size': 100,
    #          'verbose': verbose,
    #          'random_state': random_state,
    #          'training_method': training_method,
    #          'architecture': architecture,
    #          'path_model': path_model
    #          }

    n_model.add_entry(row=merge_dicts(params, {'method': method}))

    make_arguments = {key: value for key, value in params.items()
                      if key in inspect.getargspec(create_model)[0]}
    fit_arguments = {key: value for key, value in params.items()
                     if key in inspect.getargspec(create_model().fit)[0]}

    if verbose >= 1:
        pp = pprint.PrettyPrinter(indent=2)
        print(pp.pprint(create_model().get_config()))

    if file_M is None:
        M = None
    else:
        M = np.loadtxt(file_M)
        np.savetxt(os.path.join(diary.path, 'M.csv'), M)

    map_arguments = []
    skf = StratifiedKFold(n_splits=k_folds, shuffle=False)
    process_id = 0
    for i in range(n_iterations):
        X_y_s, Z_y_s, z_y_s, Y_y_s, y_y_s = shuffle(X_y, Z_y, z_y, Y_y, y_y,
                                                    random_state=i)
        splits = skf.split(X_y_s, y_y_s)

        for train, valid in splits:
            n_extra.add_entry(row={'pid': process_id, 'Counter(y_y_t)' : Counter(y_y_s[train])})
            n_extra.add_entry(row={'pid': process_id, 'y_y_t' : y_y_s[train][:5]})
            n_extra.add_entry(row={'pid': process_id, 'Y_y_t' : "\n{}".format(Y_y_s[train][:5])})
            n_extra.add_entry(row={'pid': process_id, 'Counter(z_y_t)' : Counter(z_y_s[train])})
            n_extra.add_entry(row={'pid': process_id, 'z_y_t' : z_y_s[train][:5]})
            n_extra.add_entry(row={'pid': process_id, 'Z_y_t' : "\n{}".format(Z_y_s[train][:5])})

            n_extra.add_entry(row={'pid': process_id, 'Counter(y_y_v)' : Counter(y_y_s[valid])})
            n_extra.add_entry(row={'pid': process_id, 'y_y_v' : y_y_s[valid][:5]})
            n_extra.add_entry(row={'pid': process_id, 'Y_y_v' : "\n{}".format(Y_y_s[valid][:5])})
            n_extra.add_entry(row={'pid': process_id, 'Counter(z_y_v)' : Counter(z_y_s[valid])})
            n_extra.add_entry(row={'pid': process_id, 'z_y_v' : z_y_s[valid][:5]})
            n_extra.add_entry(row={'pid': process_id, 'Z_y_v' : "\n{}".format(Z_y_s[valid][:5])})

            make_arguments['model_num'] = process_id
            classifier = MyKerasClassifier(build_fn=create_model,
                                           **make_arguments)

            parameters = [process_id, classifier,
                          X_z, Z_z,
                          X_y_s[train], Z_y_s[train], Y_y_s[train],
                          X_y_s[valid], Y_y_s[valid], fit_arguments,
                          s_diary_vars]

            if method in ['Mproper', 'EM']:
                parameters.append(M)

            map_arguments.append(tuple(parameters))

            process_id += 1

    # accuracies = train_weak_test_acc(map_arguments[0])
    if n_jobs is None or n_jobs == 1:
        my_map = map
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        my_map = pool.map

    if method == 'Mproper':
        results = my_map(train_weak_Mproper_test_results, map_arguments)
    elif method == 'fully_supervised':
        results = my_map(train_weak_fully_supervised_test_results,
                         map_arguments)
    elif method == 'fully_weak':
        results = my_map(train_weak_fully_weak_test_results, map_arguments)
    elif method in ['partially_weak', 'OSL']:
        results = my_map(train_weak_partially_weak_test_results,
                         map_arguments)
    elif method == 'EM':
        results = my_map(train_weak_EM_test_results, map_arguments)
    else:
        raise(ValueError('Method not implemented: %s' % (method)))

    dict_results = {}
    keys_results = [key for key in results[0]['history'].keys()]
    for key in keys_results:
        dict_results[key] = np.array([result['history'][key] for result in results])

    for i in range(len(results)):
        pid = results[i]['pid']
        for epoch in range(epochs):
            row = dict(pid=pid, epoch=epoch + 1)
            for key in keys_results:
                row[key] = dict_results[key][i][epoch]
            n_tra.add_entry(row=row)

    perrorevery = 0.1
    for measure in ['acc', 'loss']:
        legend = [key for key in dict_results.keys() if measure in key]
        legend.sort()
        fig = plot_errorbar([dict_results[key] for key in legend],
                            perrorevery=perrorevery, legend=legend,
                            title='{}, {}, training {}'.format(architecture,
                                method, measure))
        diary.save_figure(fig, filename=measure)

    b_v_e = dict_results['val_y_acc'].mean(axis=0).argmax() # best_validation_epoch
    cm_mean = np.zeros((n_c, n_c))
    acc_mean = 0
    for result in results:
        cm = result['history']['val_y_cm'][b_v_e]
        pid = result['pid']
        acc = np.true_divide(np.diag(cm).sum(), cm.sum())

        # Save all the results
        n_val.add_entry(row={'pid': pid, 'acc': acc, 'epoch': b_v_e,
                       'cm': cm})

        # Compute the mean of the final confusion matrix
        cm_mean += np.true_divide(cm, n_iterations)
        acc_mean += acc/len(results)

    fig = plot_confusion_matrix(cm_mean, columns=classes, rows=classes,
                                colorbar=False,
                                title='Epoch {}, Mean CM {} (acc={:.3f})'.format(
                                    b_v_e, method, acc_mean))

    diary.save_figure(fig, filename='mean_confusion_matrix')
