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

from experiments.models import MyKerasClassifier

from experiments.models import create_model

from experiments.visualizations import plot_data
from experiments.visualizations import plot_confusion_matrix
from experiments.visualizations import plot_multilabel_scatter

from experiments.diary import Diary
from experiments.utils import merge_dicts

from wlc.WLweakener import computeVirtual
from wlc.WLweakener import computeM
from wlc.WLweakener import estimate_M
from wlc.WLweakener import weak_to_index

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
            fig = plot_confusion_matrix(cm, columns=classes, rows=classes,
                                        colorbar=False,
                                        title='Confusion matrix')
            diary.save_figure(fig, filename='confusion_matrix')

    return pe_tr, pe_cv


# TODO take a look that everything is ok
def train_weak_Mproper_test_results(parameters):
    """Train a model using the Mproper approach:

        1. Learn a mixing matrix using training with weak and true labels
        2. Compute virtual labels for training set only with weak labels
        3. Train a model using the training set with virtual labels
        4. Evaluate the model in the validation set with true labels

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
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments = parameters
    n_c = Y_y_v.shape[1]
    categories = range(n_c)

    verbose = fit_arguments.get('verbose', 0)

    # 1. Learn a mixing matrix using training with weak and true labels
    M = estimate_M(Z_y_t, Y_y_t, categories, reg=None)
    # 2. Compute virtual labels for training set only with weak labels
    V_z_t = computeVirtual(Z_z_t, c=n_c, method='Mproper', M=M)
    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 3. Train a model using the training set with virtual labels and true
    #    labels
    V_t = np.concatenate((V_z_t, Y_y_t), axis=0)
    X_t = np.concatenate((X_z_t, X_y_t), axis=0)
    np.random.seed(process_id)
    X_t, V_t = shuffle(X_t, V_t)
    history = classifier.fit(X_t, V_t, **fit_arguments)
    # 4. Evaluate the model in the validation set with true labels
    # FIXME this outputs classes from 0 to #classes - 1
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    #print('MP: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history}
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
    """
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments = parameters
    n_c = Y_y_v.shape[1]
    categories = range(n_c)

    verbose = fit_arguments.get('verbose', 0)

    # 1. Learn a mixing matrix using training with weak and true labels
    M_1 = estimate_M(Z_y_t, Y_y_t, categories, reg=None)
    M_2 = computeM(c=n_c, method='supervised')
    q_1 = M_1.shape[0] / float(M_1.shape[0] + M_2.shape[0])
    q_2 = M_2.shape[0] / float(M_1.shape[0] + M_2.shape[0])
    M = np.concatenate((q_1*M_1, q_2*M_2), axis=0)
    #  2. Compute the index of each sample relating it to the corresponding
    #     row of the new mixing matrix
    #      - Needs to compute the individual M and their weight q
    Z_z_t_index = weak_to_index(Z_z_t, method='Mproper')
    Y_y_t_index = weak_to_index(Y_y_t, method='supervised')

    # 3. Give the mixing matrix to the model for future use
    #    I need to give the matrix M to the fit function
    # 4. Train model using all the sets with instead of labels the index of
    #    the corresponding rows of the mixing matrix
    Z_index_t = np.concatenate((Z_z_t_index,
                                Y_y_t_index + M_1.shape[0]))
    np.random.seed(process_id)
    X_t = np.concatenate((X_z_t, X_y_t), axis=0)
    X_t, Z_index_t = shuffle(X_t, Z_index_t)
    history = classifier.fit(X_t, Z_index_t, M=M, **fit_arguments)
    # 5. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history}
    return results


# TODO take a look that everything is ok
def train_weak_fully_supervised_test_results(parameters):
    """Train a model using the fully supervised approach:

        1. Train model with the training set that has true labels
        2. Evaluate the model in the validation set with true labels

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
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments = parameters
    n_c = Y_y_v.shape[1]
    categories = range(n_c)

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has true labels
    history = classifier.fit(X_y_t, Z_y_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    #print('FS: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history}
    return results


def train_weak_fully_weak_test_results(parameters):
    """Train a model using the fully supervised approach:

        1. Train model with the training set that has weak labels
        2. Evaluate the model in the validation set with true labels

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
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments = parameters
    n_c = Y_y_v.shape[1]
    categories = range(n_c)

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has weak labels
    X_z_t = np.concatenate([X_z_t, X_y_t])
    Z_z_t = np.concatenate([Z_z_t, Z_y_t])
    X_z_t, Z_z_t = shuffle(X_z_t, Z_z_t, random_state=process_id)
    history = classifier.fit(X_z_t, Z_z_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    #print('FW: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history}
    return results


def train_weak_partially_weak_test_results(parameters):
    """Train a model using the fully supervised approach:

        1. Train model with the training set that has weak labels with the true
           labels when available
        2. Evaluate the model in the validation set with true labels

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
    process_id, classifier, X_z_t, Z_z_t, X_y_t, Z_y_t, Y_y_t, X_y_v, Y_y_v, fit_arguments = parameters
    n_c = Y_y_v.shape[1]
    categories = range(n_c)

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has weak labels with the true
    #    labels when available
    X_z_t = np.concatenate([X_z_t, X_y_t])
    Z_z_t = np.concatenate([Z_z_t, Y_y_t])
    X_z_t, Z_z_t = shuffle(X_z_t, Z_z_t, random_state=process_id)
    history = classifier.fit(X_z_t, Z_z_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    #print('PW: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(Y_y_v, axis=1), y_pred)
    results = {'pid': process_id, 'cm': cm, 'history': history}
    return results


# TODO add other methods
def analyse_weak_labels(X_z, Z_z, z_z, X_y, Z_y, z_y, Y_y, y_y, classes,
                        n_iterations=2, k_folds=2, diary=None, verbose=0,
                        random_state=None, method='Mproper', n_jobs=None,
                        architecture='lr', loss='mse'):
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
    entry_tra = diary.add_notebook('training')

    n_s_z = X_z.shape[0]
    n_s_y = X_y.shape[0]
    n_f = X_z.shape[1]
    n_c = Y_y.shape[1]

    # If dimension is 2, we draw a 2D scatterplot
    if n_f >= 2:
        fig = plot_multilabel_scatter(X_y, Y_y, title='True labels')
        diary.save_figure(fig, filename='true_labels')

        fig = plot_multilabel_scatter(X_y, Z_y, title='Weak labels')
        diary.save_figure(fig, filename='weak_labels')


    if method == 'OSL':
        training_method = 'OSL'
    elif method in ['Mproper', 'fully_supervised', 'fully_weak',
                    'partially_weak']:
        training_method = 'supervised'
    elif method == 'EM':
        print('Training method is EM')
        training_method = 'EM'
    else:
        raise(ValueError('Method unknown {}'.format(method)))

    LOSS_MAP = dict(mse='mean_squared_error', wbs='w_brier_score',
                    bs='brier_score')

    loss = LOSS_MAP[loss]

    # Parameters for the multiprocessing training and validation
    params = {'input_dim': n_f,
              'output_size': n_c,
              'optimizer': 'rmsprop',
              'loss': loss,
              'init': 'glorot_uniform',
              'lr': 1.0,
              'momentum': 0.5,
              'decay': 0.5,
              'nesterov': True,
              'epochs': 200,
              'batch_size': 100,
              'verbose': verbose,
              'random_state': random_state,
              'training_method': training_method,
              'architecture': architecture
              }

    entry_model(row=merge_dicts(params, {'method': method}))

    make_arguments = {key: value for key, value in params.items()
                      if key in inspect.getargspec(create_model)[0]}
    fit_arguments = {key: value for key, value in params.items()
                     if key in inspect.getargspec(create_model().fit)[0]}

    classifier = MyKerasClassifier(build_fn=create_model, **make_arguments)

    if verbose >= 1:
        pp = pprint.PrettyPrinter(indent=2)
        print(pp.pprint(create_model().get_config()))

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
                                  X_y_s[test], Y_y_s[test], fit_arguments))
            process_id += 1

    #accuracies = train_weak_test_acc(map_arguments[0])
    pool = multiprocessing.Pool(processes=n_jobs)
    if method == 'Mproper':
        results = pool.map(train_weak_Mproper_test_results, map_arguments)
    elif method == 'fully_supervised':
        results = pool.map(train_weak_fully_supervised_test_results, map_arguments)
    elif method == 'fully_weak':
        results = pool.map(train_weak_fully_weak_test_results, map_arguments)
    elif method in ['partially_weak', 'OSL']:
        results = pool.map(train_weak_partially_weak_test_results, map_arguments)
    elif method == 'EM':
        results = pool.map(train_weak_EM_test_results, map_arguments)
    else:
        raise(ValueError('Method not implemented: %s' % (method)))

    # FIXME Store the results of the different epochs in csv files
    for result in results:
        if verbose > 1:
            print(result)
        history = result['history'].history
        for epoch, (loss, acc) in enumerate(zip(history['loss'], history['acc'])):
            row = dict(pid=result['pid'], epoch=epoch + 1, loss=loss, acc=acc)
            entry_tra(row=row)


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

    fig = plot_confusion_matrix(cm_mean, columns=classes, rows=classes,
                                colorbar=False,
                                title='Mean CM {} (acc={:.3f})'.format(method,
                                                                       acc_mean))
    diary.save_figure(fig, filename='mean_confusion_matrix')

