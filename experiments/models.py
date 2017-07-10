import numpy as np
import scipy as sp

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

from functools import partial

from experiments.metrics import brier_loss
from experiments.metrics import w_brier_loss


class MySequential(Sequential):
    def fit(self, train_x, train_y, test_x=None, test_y=None, batch_size=None,
            epochs=1):
        return self.model.fit(train_x, train_y, batch_size=batch_size,
                              epochs=epochs, verbose=0)


class MySequentialOSL(Sequential):
    def fit(self, train_x, train_y, test_x=None, test_y=None, batch_size=None,
            epochs=1, verbose=0):
        history = []
        for n in range(epochs):
            predictions = self.predict_proba(train_x, batch_size=batch_size)
            train_osl_y = self.hardmax(np.multiply(train_y, predictions))

            h = super(MySequentialOSL, self).fit(train_x, train_osl_y,
                                                 batch_size=batch_size,
                                                 epochs=1, verbose=verbose)
            history.append(h)
        return history

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
    def fit(self, train_x, train_y, test_x=None, test_y=None, batch_size=None,
            epochs=1, verbose=0):
        history = []
        for n in range(epochs):
            predictions = self.predict_proba(train_x, batch_size=batch_size)

            ## 'EM'  :Expected Log likelihood (i.e. the expected value of the
            ##        complete data log-likelihood after the E-step). It assumes
            ##        that a mixing matrix is known and contained in
            ##        self.params['M']
            ## Need to incorporate this part
            # Q = predictions * M[T, :].T
            # train_em_y = Q / np.sum(Q, axis=0)
            raise(ValueError('Not implemented'))

            h = super(MySequentialOSL, self).fit(train_x, train_em_y,
                                                 batch_size=batch_size,
                                                 epochs=1, verbose=verbose)
            history.append(h)
        return history

    def predict_proba(self, test_x, batch_size=None):
        return self.predict(test_x, batch_size)


def create_model(input_dim=1, output_size=1, optimizer='rmsprop',
                 init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                 nesterov=False, loss='mean_squared_error',
                 class_weights=None, training_method='supervised'):

    if training_method == 'supervised':
        model = MySequential()
    elif training_method == 'OSL':
        model = MySequentialOSL()
    elif training_method == 'EM':
        model = MySequentialEM()
    else:
        raise(ValueError('Training method %s not implemented' %
                         (training_method)))

    model.add(Dense(100, input_dim=input_dim, kernel_initializer=init,
                    activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(100, input_dim=input_dim, kernel_initializer=init,
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

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['acc'])
    return model


class KerasModel(object):
    def __init__(self, input_dim=1, output_size=1, optimizer='rmsprop',
                 init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                 nesterov=False, loss='mean_squared_error',
                 class_weights=None, random_seed=None):
        self.input_dim = input_dim
        self.output_size = output_size
        self.optimizer = optimizer
        self.random_seed = random_seed

        # TODO see why I can not initialize the seed just before I call compile
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        model = self.create_model(input_dim, output_size)

        if class_weights is None:
            self.class_weights = np.ones(output_size)
        else:
            self.class_weights = class_weights

        if optimizer in  ['sgd', 'SGD']:
            keras_opt = SGD(lr=lr, momentum=momentum, decay=decay,
                            nesterov=nesterov)
        elif optimizer == 'Adam':
            keras_opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                             decay=decay)
        elif optimizer == 'Nadam':
            keras_opt = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                              schedule_decay=decay)
        else:
            keras_opt = optimizer

        if loss == 'w_brier_score':
            loss = partial(w_brier_loss, class_weights=class_weights)
            loss.__name__ = 'w_brier_score'
        elif loss == 'brier_score':
            loss = brier_loss
            loss.__name__ = 'brier_score'

        model.compile(loss=loss, optimizer=keras_opt, metrics=['acc'])

        self.model = model

    def create_model(self, input_dim, output_size):
        model = Sequential()
        model.add(Dense(output_size, input_shape=(input_dim,)))
        model.add(Activation('softmax'))
        return model

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

    def fit(self, train_x, train_y, test_x=None, test_y=None, batch_size=None,
            nb_epoch=1):
        """
        The fit function requires both train_x and train_y.
        See 'The selected model' section above for explanation
        """
        if 'n_epoch' in self.params:
            nb_epoch = self.params['n_epoch']

        batch_size = self.batch_size if batch_size is None else batch_size

        if batch_size is None:
            batch_size = train_x.shape[0]

        # TODO try to use the OSL loss instead of iterating over epochs
        if self.OSL:
            history = []
            for n in range(nb_epoch):
                predictions = self.model.predict_proba(train_x,
                                                       batch_size=batch_size,
                                                       verbose=0)
                train_osl_y = self.hardmax(np.multiply(train_y, predictions))

                h = self.model.fit(train_x, train_osl_y, batch_size=batch_size,
                                   nb_epoch=1, verbose=0)
                history.append(h)
            return history

        return self.model.fit(train_x, train_y, batch_size=batch_size,
                              nb_epoch=nb_epoch, verbose=0)

    def predict(self, X, batch_size=None):
        # Compute posterior probability of class 1 for weights w.
        p = self.predict_proba(X, batch_size=batch_size)

        # Class
        D = np.argmax(p, axis=1)

        return D  # p, D

    def predict_proba(self, test_x, batch_size=None):
        """
        This function finds the k closest instances to the unseen test data,
        and averages the train_labels of the closest instances.
        """
        batch_size = self.batch_size if batch_size is None else batch_size

        if batch_size is None:
            batch_size = test_x.shape[0]

        return self.model.predict(test_x, batch_size=batch_size)


class KerasWeakLogisticRegression(KerasModel):
    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(output_size, input_shape=(input_size,),
                        kernel_initializer='glorot_uniform'))
        model.add(Activation('softmax'))
        return model


class KerasWeakMultilayerPerceptron(KerasModel):
    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(200, input_shape=(input_size,), kernel_initializer='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(200, kernel_initializer='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size))
        model.add(Activation('softmax'))
        return model

class KerasMultilayerPerceptron(KerasModel):
    def create_model(self, input_dim, output_size, init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(100, input_dim=input_dim, kernel_initializer=init,
                        activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(100, input_dim=input_dim, kernel_initializer=init,
                        activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, input_dim=100, kernel_initializer=init,
                        activation='softmax'))
        return model
