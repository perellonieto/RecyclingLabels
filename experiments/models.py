import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from functools import partial

from experiments.metrics import brier_loss
from experiments.metrics import w_brier_loss


def create_model(input_dim=1, output_size=1, optimizer='rmsprop',
                 init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                 nesterov=False, loss='mean_squared_error',
                 class_weights=None):
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, kernel_initializer=init,
                    activation='sigmoid'))
    model.add(Dense(output_size, input_dim=20, kernel_initializer=init,
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
