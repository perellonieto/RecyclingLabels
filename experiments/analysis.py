import sys
import time
import pprint
import inspect
import multiprocessing

from scipy import sparse
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
# TODO Change to model_selection
import sklearn.cross_validation as skcv
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier

from experiments.models import create_model

from experiments.visualizations import plot_data
from experiments.visualizations import plot_heatmap
from experiments.visualizations import plot_multilabel_scatter

from experiments.diary import Diary

from wlc.WLweakener import computeVirtual
from wlc.WLweakener import estimate_M


def n_times_k_fold_cross_val(X, V, y, classifier, n_iterations=10, k_folds=10,
                             n_jobs=-1, fit_arguments=None,
                             entry_notebook=None, classes=None, diary=None):
    """Evaluates a classifier using cross-validation

    Parameters
    ----------
    classif : object
        This is the classifier that needs to be trained and evaluated. It needs
        to have the following functions:
            - fit(X,y) :
            - predict(X) :
            - predict_proba(X) :
            - get_params() : All the necessary parameters to create a deep copy

    X : array-like, with shape (n_samples, n_dim)
        The data to fit.

    y : array-like, with shape (n_samples, )
        The target variable of integers. This array is used for the evaluation
        of the model.

    V : array-like, optional, with shape (n_samples, n_classes), default: 'y'
        The virtual target variable. This array is used for the training of the
        model.

    n_sim : integer, optional, default: 1
        The number of simulation runs.

    n_jobs : integer, optional, default: 1
        The number of CPUs to use to do the computation. -1 means 'all CPUs'

    Returns
    -------
    predictions_training : ndarray
        This are the predictions on the training set after training

    predictions_validation : ndarray
        This are the predictions on the validation set after training
    """
    n_c = V.shape[1]
    if classes is None:
        classes = [str(i) for i in range(n_c)]

    pe_cv = [0] * n_iterations

    ns = X.shape[0]
    start = time.clock()
    # ## Loop over simulation runs
    for i in xrange(n_iterations):
        X_shuff, v_shuff, y_shuff = shuffle(X, V, y, random_state=i)
        cv_start = time.clock()
        y_pred = skcv.cross_val_predict(classifier, X_shuff, v_shuff, cv=k_folds,
                                        verbose=0, n_jobs=n_jobs,
                                        fit_params=fit_arguments)
        cv_end = time.clock()

        # Estimate error rates:s
        pe_cv[i] = float(np.count_nonzero(y_shuff != y_pred)) / ns

        # ########################
        # Ground truth evaluation:
        #   Training with the given virtual labels (by default true labels)
        if i == 0:
            classifier.fit(X, V, **fit_arguments)
            f = classifier.predict_proba(X, verbose=0)

            # Then, we evaluate this classifier with all true labels
            # Note that training and test samples are being used in this error rate
            d = np.argmax(f, axis=1)
            pe_tr = float(np.count_nonzero(y != d)) / ns

        if entry_notebook is not None:
            entry_notebook(row={'pe_tr': pe_tr, 'pe_cv': pe_cv[i],
                           'cv_time': cv_end - cv_start})

        stop = time.clock()
        print(('\nAveraging {0} simulations. Estimated time to finish '
               '{1:0.4f}s.').format(n_iterations,
                                      (stop - start)/(i+1)*(n_iterations-i)))
        sys.stdout.flush()

        cm = confusion_matrix(y_shuff, y_pred)
        print("Confusion matrix: \n{}".format(cm))
        if diary is not None:
            fig = plot_heatmap(cm, columns=classes, rows=classes,
                               colorbar=False, title='Confusion matrix')
            diary.save_figure(fig, filename='confusion_matrix')

    return pe_tr, pe_cv


# TODO think about having a training and validation sets, dividing them
# randomly in k folds, and training k times with each training fold and
# validating in each of the k folds of the validation
def n_times_validation(X_train, V_train, X_val, y_val, classifier,
                       iterations=10, n_jobs=-1, fit_arguments=None,
                       entry_notebook=None, classes=None, diary=None):
    """Evaluates a classifier using

    Parameters
    ----------
    classif : object
        This is the classifier that needs to be trained and evaluated. It needs
        to have the following functions:
            - fit(X,y) :
            - predict(X) :
            - predict_proba(X) :
            - get_params() : All the necessary parameters to create a deep copy

    X : array-like, with shape (n_samples, n_dim)
        The data to fit.

    y : array-like, with shape (n_samples, )
        The target variable of integers. This array is used for the evaluation
        of the model.

    V : array-like, optional, with shape (n_samples, n_classes), default: 'y'
        The virtual target variable. This array is used for the training of the
        model.

    n_sim : integer, optional, default: 1
        The number of simulation runs.

    n_jobs : integer, optional, default: 1
        The number of CPUs to use to do the computation. -1 means 'all CPUs'

    Returns
    -------
    predictions_training : ndarray
        This are the predictions on the training set after training

    predictions_validation : ndarray
        This are the predictions on the validation set after training
    """
    n_c = V_train.shape[1]
    if classes is None:
        classes = [str(i) for i in range(n_c)]

    ns = X_train.shape[0]
    start = time.clock()
    # ## Loop over simulation runs
    for i in xrange(iterations):
        X_train_s, V_train_s = shuffle(X_train, V_train, random_state=i)
        X_val_s, y_val_s = shuffle(X_val, y_val, random_state=i)

        cv_start = time.clock()
        history = classifier.fit(X_train_s, V_train_s, **fit_arguments)
        y_pred = skcv.cross_val_predict(classifier, X_val_s, y_val_s, cv=n_folds,
                                        verbose=0, n_jobs=n_jobs,
                                        fit_params=fit_arguments)
        cv_end = time.clock()

        # Estimate error rates:s
        pe_cv[i] = float(np.count_nonzero(y_shuff != y_pred)) / ns

        # ########################
        # Ground truth evaluation:
        #   Training with the given virtual labels (by default true labels)
        if i == 0:
            classifier.fit(X, V, **fit_arguments)
            f = classifier.predict_proba(X, verbose=0)

            # Then, we evaluate this classifier with all true labels
            # Note that training and test samples are being used in this error rate
            d = np.argmax(f, axis=1)
            pe_tr = float(np.count_nonzero(y != d)) / ns

        if entry_notebook is not None:
            entry_notebook(row={'pe_tr': pe_tr, 'pe_cv': pe_cv[i],
                           'cv_time': cv_end - cv_start})

        stop = time.clock()
        print(('\nAveraging {0} simulations. Estimated time to finish '
               '{1:0.4f}s.').format(iterations,
                                      (stop - start)/(i+1)*(iterations-i)))
        sys.stdout.flush()

        cm = confusion_matrix(y_shuff, y_pred)
        print("Confusion matrix: \n{}".format(cm))
        if diary is not None:
            fig = plot_heatmap(cm, columns=classes, rows=classes,
                               colorbar=False, title='Confusion matrix')
            diary.save_figure(fig, filename='confusion_matrix')

    return pe_tr, pe_cv


def analyse_true_labels(X, Y, y, random_state=None, verbose=0, classes=None,
                        diary=None, n_jobs=None, n_iterations=2, k_folds=2):
    """ Trains a Feed-fordward neural network using cross-validation

    The training and validation is done in the validation set using the true
    labels

    Parameters
    ----------
        X: ndarray (n_samples, n_features)
        Y: ndarray (n_samples, n_classes)
            True labels in binary as a encoding one-hot encoding
        y: ndarray (n_samples, )
            True labels as integers
    """
    # Test performance on validation true labels
    # ## Create a Diary for all the logs and results
    if diary is None:
        diary = Diary(name='true_labels', path='results', overwrite=False,
                      image_format='png', fig_format='svg')

    entry_model = diary.add_notebook('model')
    entry_val = diary.add_notebook('validation')

    n_s = X.shape[0]
    n_f = X.shape[1]
    n_c = Y.shape[1]

    # If dimension is 2, we draw a scatterplot
    if n_f >= 2:
        fig = plot_data(X, y, save=False, title='True labels')
        diary.save_figure(fig, filename='true_labels')

    params = {'input_dim': n_f,
              'output_size': n_c,
              'optimizer': 'rmsprop',
              'loss': 'categorical_crossentropy',
              'init': 'glorot_uniform',
              'lr': 1.0,
              'momentum': 0.5,
              'decay': 0.5,
              'nesterov': True,
              'epochs': 100,
              'batch_size': 100,
              'verbose': verbose,
              'random_state': random_state
              }

    entry_model(row=params)

    make_arguments = {key: value for key, value in params.items()
                      if key in inspect.getargspec(create_model)[0]}
    model = KerasClassifier(build_fn=create_model, **make_arguments)
    pp = pprint.PrettyPrinter(indent=2)

    if verbose >= 1:
        print(pp.pprint(model.get_config()))

    #fit_arguments = {key: value for key, value in params.items()
    #                 if key in inspect.getargspec(model.fit)[0]}
    fit_arguments = {key: value for key, value in params.items()
                     if key in inspect.getargspec(model.build_fn().fit)[0]}

    if sparse.issparse(X):
        X = X.toarray()

    pe_tr, pe_cv = n_times_k_fold_cross_val(X=X, V=Y, y=y, classifier=model,
                                            n_iterations=n_iterations,
                                            k_folds=k_folds,
                                            n_jobs=n_jobs,
                                            fit_arguments=fit_arguments,
                                            entry_notebook=entry_val,
                                            classes=classes, diary=diary)

    #model.fit(X, Y, **fit_arguments)

    #q = model.predict_proba(X)
    #y_pred = q.argmax(axis=1)

    #acc = accuracy_score(y, y_pred)
    #print("#####")
    #print("Accuracy = {}".format(acc))
    #cm = confusion_matrix(y, y_pred)
    #print("Confusion matrix: \n{}".format(cm))
    #fig = plot_heatmap(cm, columns=classes, rows=classes, colorbar=False)
    #diary.save_figure(fig, filename='confusion_matrix')


# TODO take a look that everything is ok
def train_weak_test_results(parameters):
    """
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
    """
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v = parameters
    n_c = Y_y_v.shape[1]
    categories = range(n_c)
    M = estimate_M(Z_y_t, Y_y_t, categories, reg=None)
    V_z_t = computeVirtual(Z_z_t, c=n_c, method='Mproper', M=M)
    np.random.seed(process_id)
    classifier.fit(X_z_t, V_z_t, verbose=0, epochs=20)
    y_pred = classifier.predict(X_y_v, verbose=0)
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm}
    return results


# TODO add other methods
def analyse_weak_labels(X_z, Z_z, z_z, X_y, Z_y, z_y, Y_y, y_y, classes,
                        n_iterations=2, k_folds=2, diary=None, verbose=0,
                        random_state=None, method='Mproper', n_jobs=None):
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

    """
    # Test performance on validation true labels
    # ## Create a Diary for all the logs and results
    if diary is None:
        diary = Diary(name='weak_labels', path='results', overwrite=False,
                      image_format='png', fig_format='svg')

    entry_model = diary.add_notebook('model')
    entry_val = diary.add_notebook('validation')

    # Test for the validation error with the true labels
    #X_train, Z_train, z_train, X_val, Z_val, z_val, Y_val, y_val = load_data()

    n_s_z = X_z.shape[0]
    n_s_y = X_y.shape[0]
    n_f = X_z.shape[1]
    n_c = Y_y.shape[1]

    # If dimension is 2, we draw a scatterplot
    if n_f >= 2:
        fig = plot_multilabel_scatter(X_y, Y_y, title='True labels')
        diary.save_figure(fig, filename='true_labels')

        fig = plot_multilabel_scatter(X_y, Z_y, title='Weak labels')
        diary.save_figure(fig, filename='weak_labels')

    # Multiprocessing training and validation
    params = {'input_dim': n_f,
              'output_size': n_c,
              'optimizer': 'rmsprop',
              'loss': 'mean_squared_error',
              'init': 'glorot_uniform',
              'lr': 1.0,
              'momentum': 0.5,
              'decay': 0.5,
              'nesterov': True,
              'epochs': 50,
              'batch_size': 100,
              'verbose': verbose,
              'random_state': random_state
              }

    entry_model(row={**params, **{'method': method}})

    make_arguments = {key: value for key, value in params.items()
                      if key in inspect.getargspec(create_model)[0]}
    fit_arguments = {key: value for key, value in params.items()
                     if key not in make_arguments}

    classifier = KerasClassifier(build_fn=create_model, **make_arguments)
    pp = pprint.PrettyPrinter(indent=2)

    if verbose >= 1:
        print(pp.pprint(classifier.get_config()))

    map_arguments = []
    skf = StratifiedKFold(n_splits=k_folds, shuffle=False)
    process_id = 0
    for i in range(n_iterations):
        X_y_s, Z_y_s, z_y_s, Y_y_s, y_y_s = shuffle(X_y, Z_y, z_y, Y_y, y_y,
                                                    random_state=i)
        splits = skf.split(X_y_s, y_y_s)

        for train, test in splits:
            map_arguments.append((process_id, classifier,
                                  X_z, Z_z,
                                  X_y_s[train], Z_y_s[train], Y_y_s[train],
                                  X_y_s[test], Y_y_s[test]))
            process_id += 1

    #accuracies = train_weak_test_acc(map_arguments[0])
    pool = multiprocessing.Pool(processes=n_jobs)
    results = pool.map(train_weak_test_results, map_arguments)
    print(results)
    cm_mean = np.zeros((n_c, n_c))
    acc_mean = 0
    for result in results:
        cm = result['cm']
        pid = result['pid']
        acc = np.true_divide(np.diag(cm).sum(), cm.sum())
        entry_val(row={'pid': pid, 'acc': acc,
                       'cm': cm.__str__().replace('\n','')})
        cm_mean += np.true_divide(cm, len(results))
        acc_mean += acc/len(results)

    fig = plot_heatmap(cm_mean, columns=classes, rows=classes, colorbar=False,
                       title='Mean Confusion matrix (acc={})'.format(acc_mean))
    diary.save_figure(fig, filename='mean_confusion_matrix')

def analyse_2(load_data, random_state=None):
    """ Makes a Grid search on the hyperparameters of a Feed-fordwared neural
        network

    The training and validation is done in the validation set using the true
    labels

    Parameters
    ----------
        load_data: function
            Function that returns a training and validation set on the form
            X_train: ndarray (n_train_samples, n_features)
            Z_train: ndarray (n_train_samples, n_classes)
                Weak labels in binary as a one-hot encoding
            z_train: ndarray (n_train_samples, )
                Weak labels as integers
            X_val: ndarray (n_val_samples, n_features)
            Z_val: ndarray (n_val_samples, n_classes)
            z_val: ndarray (n_val_samples, )
            Y_val: ndarray (n_val_samples, n_classes)
                True labels in binary as a encoding one-hot encoding
            y_val: ndarray (n_val_samples, )
                True labels as integers
    """
    # ## Create a Diary for all the logs and results
    diary = Diary(name='test_1', path='results', overwrite=False,
                  image_format='png', fig_format='svg')
    diary.add_notebook('dataset')
    diary.add_notebook('model')
    diary.add_notebook('validation')

    # Test for the validation error with the true labels
    X_train, Z_train, z_train, X_val, Z_val, z_val, Y_val, y_val = load_data()

    n_s = X_train.shape[0]
    n_f = X_train.shape[1]
    n_c = Y_val.shape[1]

    print("Samples = {}\nFeatures = {}\nClasses = {}".format(n_s, n_f, n_c))
    entry_dataset(row=['n_samples', n_s, 'n_features', n_f, 'n_classes', n_c])

    # If dimension is 2, we draw a scatterplot
    if n_f >= 2:
        fig = plot_data(X_val, y_val, save=False, title='True labels')
        diary.save_figure(fig, filename='true_labels')

        fig = plot_data(X_val, z_val, save=False, title='Weak labels')
        diary.save_figure(fig, filename='weak_labels')

    # FIXME See if it is possible to add a seed to Keras model
    param_grid = {'input_dim': [n_f],
                  'output_size': [n_c],
                  'optimizer': ['sgd', 'rmsprop', 'adam', 'adadelta',
                                'adagrad', 'nadam'],
                  'loss': ['categorical_crossentropy'],
                  'init': ['glorot_uniform'],
                  'lr': [1.0],
                  'momentum': [0.5],
                  'decay': [0.5],
                  'nesterov': [True],
                  'epochs': [100],
                  'batch_size': [100],
                  'verbose': [verbose]
                  }

    entry_model(row=param_grid)

    model = KerasClassifier(build_fn=create_model, verbose=verbose)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_val, Y_val)

    print("Best: %f using %s" % (grid_result.best_score_,
                                 grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def analyse_4(load_data, random_state=None, verbose=0):
    X_train, Z_train, z_train, X_val, Z_val, z_val, Y_val, y_val = load_data()

    n_s = X_train.shape[0]
    n_f = X_train.shape[1]
    n_c = Y_val.shape[1]

    print("Samples = {}\nFeatures = {}\nClasses = {}".format(n_s, n_f, n_c))

    params = {'input_dim': n_f,
              'output_size': n_c,
              'optimizer': 'sgd',
              'loss': 'mean_squared_error',
              'init': 'glorot_uniform',
              'lr': 1.0,
              'momentum': 0.5,
              'decay': 0.5,
              'nesterov': True,
              'epochs': 20,
              'batch_size': 10,
              'verbose': verbose
              }

    model = KerasClassifier(build_fn=create_model, **params)

    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
    # It needs the initial parameters
    # FIXME train in _train and test in _val
    predictions = cross_val_predict(model, X_train, Z_train, cv=kfold)
    acc = accuracy_score(Z_train.argmax(axis=1), predictions)
    print("Accuracy = {}".format(acc))
