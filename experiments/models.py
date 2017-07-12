import numpy as np
import scipy as sp

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

from functools import partial

from experiments.metrics import brier_loss
from experiments.metrics import w_brier_loss


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
            ## Need to incorporate this part.
            ## TODO : Where do I get the M from?
            Q = predictions * M[T, :].T
            train_em_y = Q / np.sum(Q, axis=0)
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

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['acc'])
    return model
