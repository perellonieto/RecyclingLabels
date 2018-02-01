import os
import json
import numpy as np
import scipy as sp

from functools import partial
from collections import defaultdict

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.initializers import glorot_uniform
from keras.regularizers import l1_l2
import keras.backend as K

from sklearn.metrics import confusion_matrix, roc_curve, auc

from experiments.metrics import brier_loss, w_brier_loss

import copy


# TODO change all predict_proba for predict

def _merge_histories(history_list):
    dd = defaultdict(list)
    for d in history_list:
        for key, value in d.history.items():
            if not hasattr(value, '__iter__'):
                value = (value,)
            if isinstance(value, np.ndarray):
                dd[key].append(value)
            else:
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
        # y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__

        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')


def aucs(y_true, y_score):
    roc_aucs = []
    for i in range(y_true.shape[1]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_aucs.append(auc(fpr, tpr))
    return np.array(roc_aucs)

class MySequential(Sequential):
    def fit(self, train_x, train_y, batch_size=None, epochs=1, verbose=0,
            **kwargs):
        history = []
        best_epoch = 0
        for n in range(epochs):
            if verbose > 1:
                print('Epoch {} of {}'.format(n, epochs))
            h = super(MySequential, self).fit(train_x, train_y,
                                              batch_size=batch_size,
                                              epochs=1, verbose=verbose,
                                              **kwargs)
            h.history['train_y_loss'] = h.history.pop('loss')
            h.history['train_y_acc'] = h.history.pop('acc')
            h.history['val_y_loss'] = h.history.pop('val_loss')
            h.history['val_y_acc'] = h.history.pop('val_acc')
            # TODO do I need the training CM?
            prediction_y = self.predict_classes(train_x, verbose=0)
            h.history['train_y_cm'] = confusion_matrix(train_y.argmax(axis=1), prediction_y)
            if 'validation_data' in kwargs.keys():
                prediction_proba = self.predict_proba(
                        kwargs['validation_data'][0], batch_size=batch_size,
                        verbose=0)
                try:
                    h.history['val_y_auc'] = aucs(kwargs['validation_data'][1],
                                                  prediction_proba)
                except ValueError as e:
                    if np.any(np.isnan(prediction_proba)):
                        h.history['val_y_auc'] = np.zeros(train_y.shape[1])
                        pass
                    else:
                        raise e
                prediction_y = prediction_proba.argmax(axis=1)
                h.history['val_y_cm'] = confusion_matrix(
                    kwargs['validation_data'][1].argmax(axis=1),
                    prediction_y)

            history.append(h)

        return FakeHistory(_merge_histories(history))


class MySequentialWeak(Sequential):
    def fit(self, train_x, train_y, X_y_t=None, Y_y_t=None, batch_size=None,
            epochs=1, verbose=0, **kwargs):
        history = []
        for n in range(epochs):
            if verbose > 1:
                print('Epoch {} of {}'.format(n, epochs))
            h = super(MySequentialWeak, self).fit(train_x, train_y,
                                                  batch_size=batch_size,
                                                  epochs=1, verbose=verbose,
                                                  **kwargs)
            h.history['train_z_loss'] = h.history.pop('loss')
            h.history['train_z_acc'] = h.history.pop('acc')
            h.history['val_y_loss'] = h.history.pop('val_loss')
            h.history['val_y_acc'] = h.history.pop('val_acc')
            if (X_y_t is not None) and (Y_y_t is not None):
                e_loss, e_acc = self.evaluate(X_y_t, Y_y_t, verbose=verbose)
                h.history['train_y_loss'] = e_loss
                h.history['train_y_acc'] = e_acc
            # TODO do I need the training CM?
            prediction_y = self.predict_classes(X_y_t, verbose=0)
            h.history['train_y_cm'] = confusion_matrix(Y_y_t.argmax(axis=1), prediction_y)
            if 'validation_data' in kwargs.keys():
                prediction_proba = self.predict_proba(
                        kwargs['validation_data'][0],
                        batch_size=batch_size, verbose=0)
                try:
                    h.history['val_y_auc'] = aucs(kwargs['validation_data'][1],
                                                  prediction_proba)
                except ValueError as e:
                    if np.any(np.isnan(prediction_proba)):
                        h.history['val_y_auc'] = np.zeros(train_y.shape[1])
                        pass
                    else:
                        raise e
                prediction_y = prediction_proba.argmax(axis=1)
                h.history['val_y_cm'] = confusion_matrix(kwargs['validation_data'][1].argmax(axis=1), prediction_y)
            history.append(h)
        return FakeHistory(_merge_histories(history))


class MySequentialOSL(Sequential):
    def fit(self, train_x, train_y, test_x=None, test_y=None, X_y_t=None,
            Y_y_t=None, batch_size=None,
            epochs=1, verbose=0, **kwargs):
        if 'epochs' in kwargs.keys():
            kwargs.pop('epochs')
        history = []
        for n in range(epochs):
            if verbose > 1:
                print('Epoch {} of {}'.format(n, epochs))
            predictions = self.predict_proba(train_x, batch_size=batch_size)
            train_osl_y = self.hardmax(np.multiply(train_y, predictions))

            h = super(MySequentialOSL, self).fit(train_x, train_osl_y,
                                                 batch_size=batch_size,
                                                 epochs=1, verbose=verbose,
                                                 **kwargs)
            h.history['train_z_loss'] = h.history.pop('loss')
            h.history['train_z_acc'] = h.history.pop('acc')
            h.history['val_y_loss'] = h.history.pop('val_loss')
            h.history['val_y_acc'] = h.history.pop('val_acc')
            if (X_y_t is not None) and (Y_y_t is not None):
                e_loss, e_acc = self.evaluate(X_y_t, Y_y_t, verbose=verbose)
                h.history['train_y_loss'] = e_loss
                h.history['train_y_acc'] = e_acc
            # TODO do I need the training CM?
            prediction_y = self.predict_classes(X_y_t, verbose=0)
            h.history['train_y_cm'] = confusion_matrix(Y_y_t.argmax(axis=1), prediction_y)
            if 'validation_data' in kwargs.keys():
                prediction_proba = self.predict_proba(kwargs['validation_data'][0],
                                                      batch_size=batch_size, verbose=0)
                try:
                    h.history['val_y_auc'] = aucs(kwargs['validation_data'][1],
                                                  prediction_proba)
                except ValueError as e:
                    if np.any(np.isnan(prediction_proba)):
                        h.history['val_y_auc'] = np.zeros(train_y.shape[1])
                        pass
                    else:
                        raise e
                prediction_y = prediction_proba.argmax(axis=1)
                h.history['val_y_cm'] = confusion_matrix(kwargs['validation_data'][1].argmax(axis=1), prediction_y)
            history.append(h)
        return FakeHistory(_merge_histories(history))

    # FIXME batch size needs to be specified if None it brakes
    def predict_proba(self, test_x, batch_size=100, verbose=0):
        return self.predict(test_x, batch_size, verbose=verbose)


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
    def fit(self, train_x, train_t_ind, M, X_y_t=None, Y_y_t=None, test_x=None,
            test_y=None, batch_size=None, epochs=1, verbose=0, **kwargs):
        '''


        X_y_t: Samples with true labels for training
        Y_y_t: True labels of the previous samples
        '''
        history = []
        if 'epochs' in kwargs.keys():
            kwargs.pop('epochs')
        for n in range(epochs):
            if verbose > 1:
                print('Epoch {} of {}'.format(n, epochs))
            predictions = self.predict_proba(train_x, batch_size=batch_size)

            # 'EM'  :Expected Log likelihood (i.e. the expected value of the
            #        complete data log-likelihood after the E-step). It assumes
            #        that a mixing matrix is known and contained in
            #        self.params['M']
            # Need to incorporate this part.
            # train_t_ind is the index corresponding to the mixing matrix row
            Q = np.multiply(predictions, M[train_t_ind])
            # FIXME there are rows that sum to 0 and this becomes a NaN
            train_em_y = Q / np.sum(Q, axis=1).reshape(-1, 1)
            # The train_em_y are floats
            # FIXME I am training with a subsample of the data
            #   as some of the samples are zero for all the row
            index_isfinite = np.where(np.isfinite(np.sum(train_em_y, axis=1)))[0]
            h = super(MySequentialEM, self).fit(train_x[index_isfinite],
                                                train_em_y[index_isfinite],
                                                batch_size=batch_size,
                                                epochs=1, verbose=verbose,
                                                **kwargs)
            h.history['train_z_loss'] = h.history.pop('loss')
            h.history['train_z_acc'] = h.history.pop('acc')
            h.history['val_y_loss'] = h.history.pop('val_loss')
            h.history['val_y_acc'] = h.history.pop('val_acc')
            if (X_y_t is not None) and (Y_y_t is not None):
                e_loss, e_acc = self.evaluate(X_y_t, Y_y_t, verbose=verbose)
                h.history['train_y_loss'] = e_loss
                h.history['train_y_acc'] = e_acc
            # TODO do I need the training CM?
            prediction_y = self.predict_classes(X_y_t, verbose=0)
            h.history['train_y_cm'] = confusion_matrix(Y_y_t.argmax(axis=1), prediction_y)
            if 'validation_data' in kwargs.keys():
                prediction_proba = self.predict_proba(
                        kwargs['validation_data'][0], batch_size=batch_size,
                        verbose=0)
                try:
                    h.history['val_y_auc'] = aucs(kwargs['validation_data'][1],
                                                  prediction_proba)
                except ValueError as e:
                    if np.any(np.isnan(prediction_proba)):
                        h.history['val_y_auc'] = np.zeros(train_y.shape[1])
                        pass
                    else:
                        raise e
                prediction_y = prediction_proba.argmax(axis=1)
                h.history['val_y_cm'] = confusion_matrix(kwargs['validation_data'][1].argmax(axis=1), prediction_y)
            history.append(h)
        return FakeHistory(_merge_histories(history))

    def predict_proba(self, test_x, batch_size=None, verbose=0):
        if batch_size is None:
            batch_size = test_x.shape[0]
        return self.predict(test_x, batch_size, verbose=verbose)


def create_model(input_dim=1, output_size=1, optimizer='rmsprop',
                 init='glorot_uniform', lr=1, momentum=0.0, decay=0.0,
                 nesterov=False, loss='mean_squared_error',
                 class_weights=None, training_method='supervised',
                 architecture='lr', path_model=None, model_num=0, l1=0.00,
                 l2=0.001):
    """
    Parameters
    ----------
    architecture: string: lr, mlp100m, mlp100dm, mlp100ds100dm
    """

    if init == 'glorot_uniform':
        init = glorot_uniform(seed=model_num+1)

    if training_method == 'supervised':
        model = MySequential()
    elif training_method == 'weak':
        model = MySequentialWeak()
    elif training_method == 'OSL':
        model = MySequentialOSL()
    elif training_method == 'EM':
        model = MySequentialEM()
    else:
        raise(ValueError('Training method %s not implemented' %
                         (training_method)))

    reg = l1_l2(l1=l1, l2=l2)

    previous_layer = input_dim
    if architecture == 'lr':
        model.add(Dense(output_size, input_shape=(input_dim,),
                        kernel_initializer=init,
                        W_regularizer=reg,
                        activation='softmax'))
    elif architecture.startswith('mlp'):
        architecture = architecture[3:]
        while len(architecture) > 0:
            if architecture[0] == 'd':
                model.add(Dropout(0.5))
                architecture = architecture[1:]
            elif architecture[0] == 's':
                model.add(Activation('sigmoid'))
                architecture = architecture[1:]
            elif architecture[0] == 'm':
                model.add(Dense(output_size, input_dim=previous_layer,
                                kernel_initializer=init, W_regularizer=reg))
                model.add(Activation('softmax'))
                architecture = architecture[1:]
            elif architecture[0].isdigit():
                i = 1
                while len(architecture) > i and architecture[i].isdigit():
                    i += 1
                actual_layer = int(architecture[:i])
                model.add(Dense(actual_layer, input_dim=previous_layer,
                                kernel_initializer=init, W_regularizer=reg))
                architecture = architecture[i:]
                previous_layer = actual_layer
            else:
                raise(ValueError, 'Architecture with a wrong specification')
    else:
        raise(ValueError, 'Architecture with a wrong specification')

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

    # #-----------------------------------------------------------------------------------------------------------------------------------------------------
    # # AUC for a binary classifier
    # def auc(y_true, y_pred):
    #     ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    #     pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    #     pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    #     binSizes = -(pfas[1:]-pfas[:-1])
    #     s = ptas*binSizes
    #     return K.sum(s, axis=0)

    # #-----------------------------------------------------------------------------------------------------------------------------------------------------
    # # PFA, prob false alert for binary classifier
    # def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    #     y_pred = K.cast(y_pred >= threshold, 'float32')
    #     # N = total number of negative labels
    #     N = K.sum(1 - y_true)
    #     # FP = total number of false alerts, alerts from the negative class labels
    #     FP = K.sum(y_pred - y_pred * y_true)
    #     return FP/N
    # #-----------------------------------------------------------------------------------------------------------------------------------------------------
    # # P_TA prob true alerts for binary classifier
    # def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    #     y_pred = K.cast(y_pred >= threshold, 'float32')
    #     # P = total number of positive labels
    #     P = K.sum(y_true)
    #     # TP = total number of correct alerts, alerts from the positive class labels
    #     TP = K.sum(y_pred * y_true)
    #     return TP/P

    # def auc_pred(y_true, y_pred):
    #     roc_auc_list = []
    #     for i in range(y_pred.shape[1]):
    #         roc_auc_list.append(auc(K.cast(y_true == i, 'float32'), y_pred[:,i]))
    #     roc_auc = K.stack(roc_auc_list)
    #     mean_roc_auc = K.mean(roc_auc)
    #     return mean_roc_auc

    # model.compile(loss=loss, optimizer=optimizer,
    #               metrics=['acc', auc_pred])

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=['acc'])

    if path_model is not None:
        with open(os.path.join(path_model, "model_{}.json".format(model_num)), "r") as json_file:
            saved_model_json = json.loads(json_file.read())

        model_json = json.loads(model.to_json())
        # The classes are allowed to be different, as soon as the rest is equal
        saved_model_json['class_name'] = model_json['class_name']
        if model_json != saved_model_json:
            raise ValueError("The model defined and the model to load are different")
        # Get serialized weights from HDF5
        print("Loading initial weights from {}".format(os.path.join(path_model,
                                            "model_{}.h5".format(model_num))))
        model.load_weights(os.path.join(path_model,
            "model_{}.h5".format(model_num)))

    print(model.get_weights()[0][0][-10:])
    return model
