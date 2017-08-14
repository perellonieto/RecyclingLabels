import copy
import time
import numpy as np
import multiprocessing

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from keras.wrappers.scikit_learn import KerasClassifier

from wlc.WLweakener import computeM
from wlc.WLweakener import generateWeak
from wlc.WLweakener import computeVirtual
from wlc.WLweakener import binarizeWeakLabels

from experiments.models import create_model
from experiments.data import load_weak_iris


def weak_true_partition():
    # FIXME the size is unknown at the beginning
    M = computeM(3, alpha=0.8, method='quasi_IPL')
    weak_fold, true_fold, classes = load_weak_iris(M)

    X_w, Z_w, z_w = weak_fold
    X_t, Z_t, z_t, Y_t, y_t = true_fold

    n_d = X_w.shape[1]
    classes = np.unique(y_t)
    n_c = len(classes)

    V_w = computeVirtual(z_w, c=n_c, method='Mproper', M=M)

    make_arguments = {'input_dim': n_d, 'output_size': n_c}
    classifier = KerasClassifier(build_fn=create_model, **make_arguments)

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    splits = skf.split(X, y)

    # Multiprocessing training and validation
    map_arguments = []
    for train, test in splits:
        map_arguments.append((classifier, X, y_bin, train, np.array(test)))

    def train_test_acc((classifier, X, y, train, test)):
        print('Training and predicting')
        classifier = copy.deepcopy(classifier)
        classifier.fit(X[train], y[train])
        y_pred = classifier.predict(X[test])
        return np.mean(np.equal(y_pred, np.argmax(y[test], axis=1)))

    pool = multiprocessing.Pool(processes=4)
    accuracies = pool.map(train_test_acc, map_arguments)
    print(accuracies)


# Multiprocessing training and validation
def train_test_acc((process_id, classifier, X, y, train, test)):
    np.random.seed(process_id)
    classifier.fit(X[train], y[train], verbose=0, epochs=50-process_id*2)
    y_pred = classifier.predict(X[test], verbose=0)
    return np.mean(np.equal(y_pred, np.argmax(y[test], axis=1)))

def _test_multiprocessing((process_id, classifier, X, y, train, test)):
    np.random.seed(0)
    classifier.fit(X[train], y[train], verbose=0, epochs=1)
    weights = classifier.model.get_weights()
    weights[0][0][0] = process_id
    classifier.model.set_weights(weights)
    #time.sleep(np.random.rand()*3)
    #print('pid [%s] %s y[train[0]] = %s, y[test[0] = %s' % (
    #    process_id, classifier.model.name, np.argmax(y[train[0]]),
    #    np.argmax(y[test[0]])))
    #print('pid [%s] %s %s' % (
    #    process_id, classifier.model.name, classifier.model))
    print('pid [%s] %s w[0][0][0] = %s, w[-1][-1] = %s' % (
        process_id, classifier.model.name,
        classifier.model.get_weights()[0][0][0],
        classifier.model.get_weights()[-1][-1]))
    assert(process_id == classifier.model.get_weights()[0][0][0])
    return process_id


def true_labels_example(X, y):
    n_d = X.shape[1]
    classes = np.unique(y)
    n_c = len(classes)

    Y = label_binarize(y, classes)

    # Multiprocessing training and validation
    make_arguments = {'input_dim': n_d, 'output_size': n_c}
    classifier = KerasClassifier(build_fn=create_model, **make_arguments)

    map_arguments = []
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    process_id = 0
    for i in range(5):
        X_shuff, y_shuff, Y_shuff = shuffle(X, y, Y, random_state=i)
        splits = skf.split(X_shuff, y_shuff)

        for train, test in splits:
            map_arguments.append((process_id, classifier,
                                  X_shuff, Y_shuff, train, np.array(test)))
            process_id += 1

    pool = multiprocessing.Pool(processes=None)
    accuracies = pool.map(train_test_acc, map_arguments)
    print(accuracies)
    print('mean accuracy = %s' % (np.mean(accuracies)))


if __name__ == '__main__':
    #dataset = load_iris()
    dataset = load_digits()
    X = dataset.data
    y = dataset.target

    start = time.time()
    true_labels_example(X, y)
    end = time.time()
    print('%s seconds' % (end-start))
