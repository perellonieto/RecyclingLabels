import numpy as np
import scipy as sp

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

from functools import partial

from experiments.metrics import brier_loss
from experiments.metrics import w_brier_loss

from keras.wrappers.scikit_learn import KerasClassifier

from collections import defaultdict

def _merge_histories(history_list):
    dd = defaultdict(list)
    for d in history_list:
        for key, value in d.history.items():
            if not hasattr(value, '__iter__'):
                value = (value,)
            [dd[key].append(v) for v in value]
    return dict(dd)


class FakeHistory(object):
    def __init__(self, history):
        self.history = history


class MyKerasClassifier(KerasClassifier):
    """This is a modification of the KerasClassifier in order to keep the
    labels with the original values.

    Implementation of the scikit-learn classifier API for Keras.
    """

    # I change the function fit to avoid any modification of the labels
    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where n_samples in the number of samples
                and n_features is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for X.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`

        # Returns
            history : object
                details about the training history at each epoch.

        # Raises
            ValueError: In case of invalid shape for `y` argument.
        """
        y = np.array(y)
        return super(KerasClassifier, self).fit(x, y, **kwargs)

    # Now it does not modify the predicted labels
    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        classes = self.model.predict_classes(x, **kwargs)
        return classes

    # Now it does not modify the labels
    def score(self, x, y, **kwargs):
        """Returns the mean accuracy on the given test data and labels.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where n_samples in the number of samples
                and n_features is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for x.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        # Returns
            score: float
                Mean accuracy of predictions on X wrt. y.

        # Raises
            ValueError: If the underlying model isn't configured to
                compute accuracy. You should pass `metrics=["accuracy"]` to
                the `.compile()` method of the model.
        """
        #y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')

class MySequentialOSL(Sequential):
    def fit(self, train_x, train_y, test_x=None, test_y=None, batch_size=None,
            epochs=1, verbose=0):
        history = []
        for n in range(epochs):
            if verbose > 1:
                print('Epoch {} of {}'.format(n, epochs))
            predictions = self.predict_proba(train_x, batch_size=batch_size)
            train_osl_y = self.hardmax(np.multiply(train_y, predictions))

            h = super(MySequentialOSL, self).fit(train_x, train_osl_y,
                                                 batch_size=batch_size,
                                                 epochs=1, verbose=verbose)
            history.append(h)
        return FakeHistory(_merge_histories(history))

    def predict_proba(self, test_x, batch_size=None):
        return self.predict(test_x, batch_size)

    def hardmax(self, Z):
        """ Transform each row in array Z into another row with zeroes in the
            non-maximum values and 1/nmax on the maximum values, where nmax is
            the number of elements taking the maximum value
        """
        D = sp.equal(Z, np.max(Z, axis=1, keepdims=True))

        # In case more than one value is equal to the maximum, the output
        # of hardmax is nonzero for all of them, but normalized
        D = D/np.sum(D, axis=1, keepdims=True)

        return D


class MySequentialEM(Sequential):
    def fit(self, train_x, train_t_ind, M, test_x=None, test_y=None,
            batch_size=None, epochs=1, verbose=0):
        history = []
        for n in range(epochs):
            if verbose > 1:
                print('Epoch {} of {}'.format(n, epochs))
            predictions = self.predict_proba(train_x, batch_size=batch_size)

            ## 'EM'  :Expected Log likelihood (i.e. the expected value of the
            ##        complete data log-likelihood after the E-step). It assumes
            ##        that a mixing matrix is known and contained in
            ##        self.params['M']
            ## Need to incorporate this part.
            # train_t_ind is the index corresponding to the mixing matrix row
            Q = np.multiply(predictions, M[train_t_ind])
            # FIXME there are rows that sum to 0 and this becomes a NaN
            train_em_y = Q / np.sum(Q, axis=1)
            # The train_em_y are floats
            # FIXME I am training with a subsample of the data
            #   as some of the samples are zero for all the row
            index_isfinite = np.where(np.isfinite(np.sum(train_em_y, axis=1)))[0]
            h = super(MySequentialEM, self).fit(train_x[index_isfinite],
                                                train_em_y[index_isfinite],
                                                batch_size=batch_size,
                                                epochs=1, verbose=verbose)
            history.append(h)
        return FakeHistory(_merge_histories(history))

    def predict_proba(self, test_x, batch_size=None):
        if batch_size is None:
            batch_size = test_x.shape[0]
        return self.predict(test_x, batch_size)


def create_model(input_dim=1, output_size=1, optimizer='rmsprop',
                 init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                 nesterov=False, loss='mean_squared_error',
                 class_weights=None, training_method='supervised',
                 architecture='lr'):
    """
    Parameters
    ----------
    architecture: string: lr, mlp100, mlp100d, mlp100d100d
    """

    if training_method == 'supervised':
        model = Sequential()
    elif training_method == 'OSL':
        model = MySequentialOSL()
    elif training_method == 'EM':
        model = MySequentialEM()
    else:
        raise(ValueError('Training method %s not implemented' %
                         (training_method)))

    if architecture == 'lr':
        model.add(Dense(output_size, input_shape=(input_dim,),
                        kernel_initializer='glorot_uniform',
                        activation='softmax'))
    elif architecture == 'mlp100':
        model.add(Dense(100, input_dim=input_dim, kernel_initializer=init,
                        activation='sigmoid'))
        model.add(Dense(output_size, input_dim=100, kernel_initializer=init,
                        activation='softmax'))
    elif architecture == 'mlp100d':
        model.add(Dense(100, input_dim=input_dim, kernel_initializer=init,
                        activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, input_dim=100, kernel_initializer=init,
                        activation='softmax'))
    elif architecture == 'mlp100d100d':
        model.add(Dense(100, input_dim=input_dim, kernel_initializer=init,
                        activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(100, input_dim=100, kernel_initializer=init,
                        activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, input_dim=100, kernel_initializer=init,
                        activation='softmax'))

    if optimizer == 'sgd':
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay,
                        nesterov=nesterov)

    if class_weights is None:
        class_weights = np.ones(output_size)

    if loss == 'w_brier_score':
        loss = partial(w_brier_loss, class_weights=class_weights)
        loss.__name__ = 'w_brier_score'
    elif loss == 'brier_score':
        loss = brier_loss
        loss.__name__ = 'brier_score'
    elif loss == 'mean_squared_error':
        loss = 'mean_squared_error'
    else:
        raise(ValueError('Unknown loss: {}'.format(loss)))

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['acc'])
    return model
