import pprint
import inspect
import multiprocessing

import numpy as np

from sklearn.model_selection import StratifiedKFold
# TODO Change to model_selection
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from experiments.models import create_model, MyKerasClassifier
from experiments.visualizations import plot_confusion_matrix, \
                                       plot_multilabel_scatter, \
                                       plot_errorbar
from experiments.diary import Diary
from experiments.utils import merge_dicts

from wlc.WLweakener import computeVirtual, computeM, estimate_M, weak_to_index


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
    # Add validation results during training
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_t, V_t, **fit_arguments)
    # 4. Evaluate the model in the validation set with true labels
    # FIXME this outputs classes from 0 to #classes - 1
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('MP: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
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
    # Add validation results during training
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
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

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has true labels
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_y_t, Z_y_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('FS: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
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

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has weak labels
    X_z_t = np.concatenate([X_z_t, X_y_t])
    Z_z_t = np.concatenate([Z_z_t, Z_y_t])
    X_z_t, Z_z_t = shuffle(X_z_t, Z_z_t, random_state=process_id)
    # Add validation results during training
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_z_t, Z_z_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('FW: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
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

    verbose = fit_arguments.get('verbose', 0)

    # TODO where is the randomization applied?
    np.random.seed(process_id)
    # 1. Train model with the training set that has weak labels with the true
    #    labels when available
    X_z_t = np.concatenate([X_z_t, X_y_t])
    Z_z_t = np.concatenate([Z_z_t, Y_y_t])
    X_z_t, Z_z_t = shuffle(X_z_t, Z_z_t, random_state=process_id)
    fit_arguments['validation_data'] = (X_y_v, Y_y_v)
    history = classifier.fit(X_z_t, Z_z_t, **fit_arguments)
    # 2. Evaluate the model in the validation set with true labels
    y_pred = classifier.predict(X_y_v, verbose=verbose)
    # print('PW: predictions min: {}, max: {}'.format(min(y_pred), max(y_pred)))
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

    # accuracies = train_weak_test_acc(map_arguments[0])
    pool = multiprocessing.Pool(processes=n_jobs)
    if method == 'Mproper':
        results = pool.map(train_weak_Mproper_test_results, map_arguments)
    elif method == 'fully_supervised':
        results = pool.map(train_weak_fully_supervised_test_results,
                           map_arguments)
    elif method == 'fully_weak':
        results = pool.map(train_weak_fully_weak_test_results, map_arguments)
    elif method in ['partially_weak', 'OSL']:
        results = pool.map(train_weak_partially_weak_test_results,
                           map_arguments)
    elif method == 'EM':
        results = pool.map(train_weak_EM_test_results, map_arguments)
    else:
        raise(ValueError('Method not implemented: %s' % (method)))

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for result in results:
        train_acc.append([])
        train_loss.append([])
        val_acc.append([])
        val_loss.append([])
        if verbose > 1:
            print(result)
        history = result['history'].history
        if 'val_loss' in history.keys():
            zipped = zip(history['loss'], history['acc'],
                         history['val_loss'], history['val_acc'])
            for epoch, (t_loss, t_acc, v_loss, v_acc) in enumerate(zipped):
                train_acc[-1].append(t_acc)
                train_loss[-1].append(t_loss)
                val_acc[-1].append(v_acc)
                val_loss[-1].append(v_loss)
                row = dict(pid=result['pid'], epoch=epoch + 1, loss=t_loss,
                           acc=t_acc, val_loss=v_loss, val_acc=v_acc)
                entry_tra(row=row)
        else:
            zipped = zip(history['loss'], history['acc'])
            for epoch, (t_loss, t_acc) in enumerate(zipped):
                train_acc[-1].append(t_acc)
                train_loss[-1].append(t_loss)
                row = dict(pid=result['pid'], epoch=epoch + 1, loss=t_loss,
                           acc=t_acc)
                entry_tra(row=row)

    train_acc = np.array(train_acc)
    train_loss = np.array(train_loss)
    val_acc = np.array(val_acc)
    val_loss = np.array(val_loss)

    if val_acc.shape[1] > 0:
        fig1 = plot_errorbar([train_acc, val_acc], errorevery=0.1,
                             title='{}, {}, training acc'.format(
                                architecture, method),
                             legend=['train', 'val'])

        fig2 = plot_errorbar([train_loss, val_loss], errorevery=0.1,
                             title='{}, {}, training loss ({})'.format(
                                architecture, method, loss),
                             legend=['train', 'val'])
    else:
        fig1 = plot_errorbar(train_acc, errorevery=0.1,
                             title='{}, {}, training acc'.format(
                                architecture, method))

        fig2 = plot_errorbar(train_loss, errorevery=0.1,
                             title='{}, {}, training loss ({})'.format(
                                architecture, method, loss))

    diary.save_figure(fig1, filename='training_accuracy')
    diary.save_figure(fig2, filename='training_loss')

    cm_mean = np.zeros((n_c, n_c))
    acc_mean = 0
    for result in results:
        cm = result['cm']
        pid = result['pid']
        acc = np.true_divide(np.diag(cm).sum(), cm.sum())
        entry_val(row={'pid': pid, 'acc': acc,
                       'cm': cm.__str__().replace('\n', '')})
        cm_mean += np.true_divide(cm, len(results))
        acc_mean += acc/len(results)

    fig = plot_confusion_matrix(cm_mean, columns=classes, rows=classes,
                                colorbar=False,
                                title='Mean CM {} (acc={:.3f})'.format(method,
                                                                       acc_mean))
    diary.save_figure(fig, filename='mean_confusion_matrix')
