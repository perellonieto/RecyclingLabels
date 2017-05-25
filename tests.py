import pprint
import inspect

from scipy import sparse

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier

from testUtils import create_model
from testUtils import plot_data
from testUtils import plot_heatmap

from testData import load_toy_example
from testData import load_blobs
from testData import load_webs

from diary import Diary

seed = 42

def test_1(load_data):
    """ Trains a Feed-fordward neural network using cross-validation

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
    # Test performance on validation true labels
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
    diary.add_entry('dataset', ['n_samples', n_s, 'n_features', n_f,
                                'n_classes', n_c])

    # If dimension is 2, we draw a scatterplot
    if n_f >= 2:
        fig = plot_data(X_val, y_val, save=False, title='True labels')
        diary.save_figure(fig, filename='true_labels')

        fig = plot_data(X_val, z_val, save=False, title='Weak labels')
        diary.save_figure(fig, filename='weak_labels')


    params = {'input_dim': n_f,
              'output_size': n_c,
              'optimizer': 'rmsprop', # 'adadelta' 'adagrad' 'sgd' 'rmsprop'
              # 'brier_score' 'w_brier_score' 'categorical_crossentropy' 'mean_squared_error'
              'loss': 'categorical_crossentropy',
              'init': 'glorot_uniform',
              'lr': 1.0,
              'momentum': 0.5,
              'decay': 0.5,
              'nesterov': True,
              'epochs': 100,
              'batch_size': 100,
              'verbose': 1,
              'seed': seed
              }

    diary.add_entry('model', params)

    make_arguments = {key: value for key, value in params.iteritems()
                                if key in inspect.getargspec(create_model)[0]}
    model = create_model(**make_arguments)
    pp = pprint.PrettyPrinter(indent=2)
    print pp.pprint(model.get_config())

    fit_arguments = {key: value for key, value in params.iteritems()
                               if key in inspect.getargspec(model.fit)[0]}

    if sparse.issparse(X_val):
        X_val = X_val.toarray()

    model.fit(X_val, Y_val, **fit_arguments)

    q = model.predict_proba(X_val)
    y_pred = q.argmax(axis=1)

    acc = accuracy_score(y_val, y_pred)
    print("#####")
    print("Accuracy = {}".format(acc))
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix: \n{}".format(cm))


def test_2(load_data):
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
    diary.add_entry('dataset', ['n_samples', n_s, 'n_features', n_f,
                                'n_classes', n_c])

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
                      'adagrad', 'nadam'], # 'adadelta' 'adagrad' 'sgd' 'rmsprop'
                  # 'brier_score' 'w_brier_score' 'categorical_crossentropy' 'mean_squared_error'
                  'loss': ['categorical_crossentropy'], #, 'mean_squared_error',
                          # 'brier_score', 'w_brier_score'],
                  'init': ['glorot_uniform'],
                  'lr': [1.0],
                  'momentum': [0.5],
                  'decay': [0.5],
                  'nesterov': [True],
                  'epochs': [100],
                  'batch_size': [100],
                  'verbose': [1]
                  }

    diary.add_entry('model', param_grid)

    model = KerasClassifier(build_fn=create_model, verbose=0)

    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_val, Y_val)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def test_3(load_data):
    """ Trains a Feed-fordward neural network using cross-validation

    The training is done with the weak labels on the training set and
    the model is evaluated with the true labels in the validation set

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
    # Test performance on validation true labels
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
    diary.add_entry('dataset', ['n_samples', n_s, 'n_features', n_f,
                                'n_classes', n_c])

    # If dimension is 2, we draw a scatterplot
    if n_f >= 2:
        fig = plot_data(X_val, y_val, save=False, title='True labels')
        diary.save_figure(fig, filename='true_labels')

        fig = plot_data(X_val, z_val, save=False, title='Weak labels')
        diary.save_figure(fig, filename='weak_labels')


    params = {'input_dim': n_f,
              'output_size': n_c,
              'optimizer': 'rmsprop', # 'adadelta' 'adagrad' 'sgd' 'rmsprop'
              # 'brier_score' 'w_brier_score' 'categorical_crossentropy' 'mean_squared_error'
              'loss': 'categorical_crossentropy',
              'init': 'glorot_uniform',
              'lr': 1.0,
              'momentum': 0.5,
              'decay': 0.5,
              'nesterov': True,
              'epochs': 100,
              'batch_size': 100,
              'verbose': 1,
              'seed': seed
              }

    diary.add_entry('model', params)

    make_arguments = {key: value for key, value in params.iteritems()
                                if key in inspect.getargspec(create_model)[0]}
    model = create_model(**make_arguments)
    pp = pprint.PrettyPrinter(indent=2)
    print pp.pprint(model.get_config())

    fit_arguments = {key: value for key, value in params.iteritems()
                               if key in inspect.getargspec(model.fit)[0]}

    if sparse.issparse(X_train):
        X_train = X_train.toarray()
        Z_train = Z_train
        z_train = z_train
        X_val = X_val.toarray()

    model.fit(X_train, Z_train, **fit_arguments)

    q = model.predict_proba(X_val)
    y_pred = q.argmax(axis=1)

    acc = accuracy_score(y_val, y_pred)
    print("#####")
    print("Accuracy = {}".format(acc))
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix: \n{}".format(cm))
    fig = plot_heatmap(cm, title='Confusion matrix')
    diary.save_figure(fig, filename='confusion_matrix')


def test_4(load_data):
    X_train, Z_train, z_train, X_val, Z_val, z_val, Y_val, y_val = load_data()

    seed = 42
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
             'verbose': 0
             }

    model = KerasClassifier(build_fn=create_model, **params)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    # It needs the initial parameters
    # FIXME train in _train and test in _val
    predictions = cross_val_predict(model, X_train, Z_train, cv=kfold)
    acc = accuracy_score(Z_train.argmax(axis=1), predictions)
    print("Accuracy = {}".format(acc))


def test_1a():
    # Working
    test_1(load_toy_example)


def test_1b():
    test_1(load_blobs)


def test_1c():
    test_1(load_webs)


def test_2a():
    # Working
    test_2(load_toy_example)


def test_2b():
    # Working
    test_2(load_blobs)


def test_3a():
    # Working
    test_3(load_toy_example)


def test_3b():
    test_3(load_blobs)


def test_3c():
    test_3(load_webs)

if __name__ == '__main__':
    test_3c()
    #main()
