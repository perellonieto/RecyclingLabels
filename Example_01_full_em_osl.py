import glob
import os
import sys
import errno
import argparse
import numpy
import keras
from keras import backend as K
from keras import regularizers

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import sklearn.datasets as datasets
from wlc.WLweakener import computeM, generateWeak, weak_to_index, binarizeWeakLabels
from experiments.visualizations import plot_history
from experiments.visualizations import plot_multilabel_scatter

import pandas

cmap = plt.cm.get_cmap('Accent')

from cycler import cycler
default_cycler = (cycler(color=['darkred', 'forestgreen', 'darkblue', 'violet', 'darkorange', 'saddlebrown']) +
                  cycler(linestyle=['-', '--', '-.', '-', '--', '-.']) + 
                  cycler(marker=['o', 'v', 'x', '*', '+', '.']) +
                  cycler(lw=[2, 1.8, 1.6, 1.4, 1.2, 1]))

plt.rcParams['figure.figsize'] = (3, 2)
plt.rcParams["figure.dpi"] = 100
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=default_cycler)


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Exampla 01''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--m-method', dest='m_method', type=str,
                        default='noisy',
                        help='''Mixing process type: noisy, random_noise,
                        random_weak, IPL, quasi_IPL''')
    parser.add_argument('-o', '--out-folder', dest='output_folder', type=str,
                        default='results',
                        help='''Folder to save the results''')
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str,
                        default='diagonals',
                        help='''Dataset to use for evaluation: diagonals,
                        6blobs''')
    parser.add_argument('-b', '--beta', dest='beta', type=float,
                        default=0.0, help='Parameter beta for the mixing process')
    parser.add_argument('-r', '--random-state', dest='random_state', type=int,
                        default=42, help='Seed for the random number generator')
    parser.add_argument('-t', '--train-proportion', dest='train_proportion',
                        type=float, default=1.0,
                        help='Proportion of training data to keep')
    parser.add_argument('-e', '--max-epochs', dest='max_epochs',
                        type=int, default=1000,
                        help='Maximum number of epochs')
    parser.add_argument('--redirect-std', dest='redirect_std',
                        action='store_true',
                        help='Redirect standard output and error')
    return parser.parse_args()


def load_dataset(dataset_name, random_state):
    if dataset_name == 'diagonals':
        n_samples = 10000
        n_classes = 6
        numpy.random.seed(random_state)
        X = numpy.random.randn(n_samples, 2)
        y = numpy.random.randint(0, n_classes, n_samples)
        X += y.reshape(-1,1)
    elif dataset_name == '6blobs':
        n_samples = 10000
        n_classes = 6
        X, y = datasets.make_blobs(n_samples=n_samples, centers=n_classes,
                                   n_features=2, random_state=random_state)
    else:
        raise ValueError('Dataset name {} not available'.format(dataset_name))
    Y = label_binarize(y, range(n_classes))
    return X, Y, y

def generate_summary(dataset_name, output_folder):
    files_list = glob.glob(output_folder + "/" + dataset_name +  "*summary.csv")

    list_ = []

    for file_ in files_list:
        df = pandas.read_csv(file_,index_col=0, header=None).T
        list_.append(df)

    df = pandas.concat(list_, axis = 0, ignore_index = True)
    df = df[df['dataset_name'] == dataset_name]
    del df['dataset_name']
    df_grouped = df.groupby(['beta', 'm_method'])
    for name, df_ in df_grouped:
        print(name)
        n_iterations = len(df_['random_state'].unique())
        df_ = df_.drop(columns=['beta', 'm_method', 'random_state'])
        df_ = df_.apply(pandas.to_numeric)
        df_.index = df_['last_train_index']
        del df_['last_train_index']
        df_.sort_index(inplace=True)
        df_ = df_.groupby(df_.index).mean()
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        for column in df_.columns:
            ax.plot(df_.index, df_[column], label=column)
        ax.set_title('dataset {}\nM = {}, beta = {}'.format(dataset_name, name[1], name[0]))
        ax.set_ylabel('Mean acc. (#it {})'.format(n_iterations))
        ax.set_xlabel('Number of weak samples')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder,
                                 '{}_{}_b{:02.0f}.svg'.format(dataset_name,
                                                             name[1],
                                                             float(name[0])*10)))

def main(dataset_name, m_method, beta, random_state, train_proportion, output_folder,
         max_epochs, redirect_std):
    try:
        os.makedirs(output_folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    unique_id = '{}_{}_b{:02.0f}_prop{:03.0f}_r{}'.format(dataset_name,
                                                          m_method, beta*10,
                                                          train_proportion*100,
                                                          random_state)
    unique_file = os.path.join(output_folder, unique_id)

    if redirect_std:
        sys.stdout = open(unique_file + '.out', 'w')
        sys.stderr = open(unique_file + '.err', 'w')

    X, Y, y = load_dataset(dataset_name, random_state)
    n_samples = X.shape[0]
    n_classes = Y.shape[1]

    M_weak = computeM(n_classes, alpha=(1-beta), beta=beta, method=m_method,
            seed=random_state)
    if M_weak.shape[0] == 2**M_weak.shape[1]:
        M_weak[0,:] = 0
        M_weak /= M_weak.sum(axis=0)
    print(numpy.round(M_weak, decimals=3))
    z = generateWeak(y, M_weak, seed=random_state)
    Z = binarizeWeakLabels(z, c=n_classes)

    M_weak_indices = weak_to_index(Z, method=m_method)
    V_weak = M_weak[M_weak_indices]

    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(1, 3, 1)
    _ = plot_multilabel_scatter(X[:100], Y[:100], fig=fig,
                                ax=ax, title='True labels', cmap=cmap)
    ax = fig.add_subplot(1, 3, 2)
    _ = plot_multilabel_scatter(X[:100], Z[:100], fig=fig,
                                ax=ax, title='Weak labels', cmap=cmap)
    ax = fig.add_subplot(1, 3, 3)
    _ = plot_multilabel_scatter(X[:100], V_weak[:100], fig=fig,
                                ax=ax, title='M rows', cmap=cmap)
    fig.savefig(unique_file + '_sample.svg')
    plt.close(fig)

    # # Divide into training, validation and test
    X_weak_train, X_true_train, X_val, X_test = numpy.array_split(X, 4)
    Y_weak_train, Y_true_train, Y_val, Y_test = numpy.array_split(Y, 4)
    Z_weak_train, Z_true_train, Z_val, Z_test = numpy.array_split(Z, 4)
    V_weak_train, V_true_train, V_val, V_test = numpy.array_split(V_weak, 4)
    y_weak_train, y_true_train, y_val, y_test = numpy.array_split(y, 4)

    # Remove a portion of the weak data
    last_train_index = int(numpy.ceil(train_proportion*X_weak_train.shape[0]))
    X_weak_train = X_weak_train[:last_train_index]
    Y_weak_train = Y_weak_train[:last_train_index]
    Z_weak_train = Z_weak_train[:last_train_index]
    V_weak_train = V_weak_train[:last_train_index]
    y_weak_train = y_weak_train[:last_train_index]

    # Save the final model for each method
    final_models = {}

    # # Define a common model
    from keras.callbacks import EarlyStopping, Callback

    def log_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        out = -y_true*K.log(y_pred)
        return K.mean(out, axis=-1)

    # Callback to show performance per epoch in the same line
    class EpochCallback(Callback):
        def on_epoch_end(self, epoch, logs={}):
            print('\rEpoch {}, val_loss = {:.2e}, val_acc = {:.2f}'.format(epoch, logs['val_loss'], logs['val_acc']), end=' ')

    # Callback for early stopping
    epoch_callback = EpochCallback()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=int(max_epochs/20),
                                   verbose=2, mode='auto', baseline=None,
                                   restore_best_weights=True)

    def make_model(loss):
        # Careful that it is ussing global variables for the input and output shapes
        numpy.random.seed(random_state)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(Y.shape[1], input_dim=X.shape[1],
                                     kernel_regularizer=regularizers.l2(0.0),
                                     activation='softmax'))
        model.compile(optimizer='adam', loss=loss, metrics=['acc', 'mse', 'ce'])
        return model

    # Keyword arguments for the fit function
    fit_kwargs = dict(validation_data=(X_val, Y_val), epochs=max_epochs, verbose=0,
                      callbacks=[early_stopping, epoch_callback], shuffle=True)


    # # Fully supervised (upperbound)
    #
    # Train with all true labels
    train_method = 'Supervised'
    print('Training ' + train_method)

    model = make_model(log_loss)

    history = model.fit(numpy.concatenate([X_weak_train, X_true_train]),
                        numpy.concatenate([Y_weak_train, Y_true_train]),
                        **fit_kwargs)

    fig = plot_history(history, model, X_test, y_test)
    fig.savefig(unique_file + '_' + train_method + '.svg')
    plt.close(fig)

    final_models[train_method] = model


    # # Our method with EM and original M
    #
    # Train EM with all weak labels

    # In[7]:
    train_method = 'EMorigM'
    print('Training ' + train_method)


    def EM_log_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        Q = y_true * y_pred
        Z_em_train = Q / K.sum(Q, axis=-1, keepdims=True)
        out = -K.stop_gradient(Z_em_train)*K.log(y_pred)
        return K.mean(out, axis=-1)

    model = make_model(EM_log_loss)

    M_true = computeM(n_classes, method='supervised')
    q_weak = X_weak_train.shape[0] / (X_weak_train.shape[0] + X_true_train.shape[0])
    q_true = X_true_train.shape[0] / (X_weak_train.shape[0] + X_true_train.shape[0])
    M = numpy.concatenate((q_weak*M_weak, q_true*M_true))

    M_true_indices = weak_to_index(Y_true_train, method='supervised') + M_weak.shape[0]
    V_true_train = M[M_true_indices]

    history = model.fit(numpy.concatenate([X_weak_train, X_true_train]),
                        numpy.concatenate([V_weak_train, V_true_train]),
                        **fit_kwargs)

    fig = plot_history(history, model, X_test, y_test)
    fig.savefig(unique_file + '_' + train_method + '.svg')
    plt.close(fig)

    final_models[train_method] = model


    # # Our method with EM and estimated M

    # In[8]:
    train_method = 'EMestM'
    print('Training ' + train_method)

    from wlc.WLweakener import estimate_M

    M_estimated = estimate_M(Z_true_train, Y_true_train, range(n_classes), reg='Partial', Z_reg=Z_weak_train)
    M_true = computeM(n_classes, method='supervised')

    q_estimated = X_weak_train.shape[0] / (X_weak_train.shape[0] + X_true_train.shape[0])
    q_true = X_true_train.shape[0] / (X_weak_train.shape[0] + X_true_train.shape[0])
    M = numpy.concatenate((q_estimated*M_estimated, q_true*M_true))

    M_estimated_indices = weak_to_index(Z_weak_train, method='random_weak')
    V_weak_train = M_estimated[M_estimated_indices]

    M_true_indices = weak_to_index(Y_true_train, method='supervised') + M_estimated.shape[0]
    V_true_train = M[M_true_indices]

    model = make_model(EM_log_loss)
    history = model.fit(numpy.concatenate([X_weak_train, X_true_train]),
                        numpy.concatenate([V_weak_train, V_true_train]),
                        **fit_kwargs)

    fig = plot_history(history, model, X_test, y_test)
    fig.savefig(unique_file + '_' + train_method + '.svg')
    plt.close(fig)

    final_models[train_method] = model

    # # Weak (lowerbound)

    # In[9]:
    train_method = 'Weak'
    print('Training ' + train_method)


    model = make_model(log_loss)
    history = model.fit(numpy.concatenate([X_weak_train, X_true_train]),
                        numpy.concatenate([Z_weak_train, Y_true_train]),
                        **fit_kwargs)
    fig = plot_history(history, model, X_test, y_test)
    fig.savefig(unique_file + '_' + train_method + '.svg')
    plt.close(fig)

    final_models[train_method] = model


    # In[10]:
    train_method = 'OSL'
    print('Training ' + train_method)


    def OSL_log_loss(y_true, y_pred):
        # Careful, I had to use a global variable here for the number of classes
        # for some reason I can not use y_osl.shape[-1] in the reshape function
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        y_osl = y_true * y_pred
        y_osl_max = K.max(y_osl, axis=-1)
        y_osl_max = K.repeat_elements(y_osl_max, n_classes, 0)
        y_osl_max = K.reshape(y_osl_max, (-1, n_classes))
        y_osl = K.cast(K.equal(y_osl, y_osl_max), y_pred.dtype)
        y_osl = y_osl / K.sum(y_osl, axis=-1, keepdims=True)
        out = -K.stop_gradient(y_osl) * K.log(y_pred)
        return K.mean(out, axis=-1)

    model = make_model(OSL_log_loss)
    history = model.fit(numpy.concatenate([X_weak_train, X_true_train]),
                        numpy.concatenate([Z_weak_train, Y_true_train]),
                        **fit_kwargs)
    fig = plot_history(history, model, X_test, y_test)
    fig.savefig(unique_file + '_' + train_method + '.svg')
    plt.close(fig)

    final_models[train_method] = model

    fig = plt.figure(figsize=(15, 4))
    lowest_acc = 1.0
    highest_acc = 0.0
    test_acc_dict = {}
    for i, (key, model) in enumerate(sorted(final_models.items())):
        lw = (len(final_models)+5 - i)/5
        p = plt.plot(model.history.history['val_acc'], lw=lw, label='Val. ' + key)
        test_acc = numpy.mean(model.predict_classes(X_test) == y_test)
        plt.axhline(y=test_acc, color=p[0].get_color(), lw=lw, linestyle='--')
        lowest_acc = test_acc if test_acc < lowest_acc else lowest_acc
        highest_acc = test_acc if test_acc > highest_acc else highest_acc
        test_acc_dict[key] = test_acc
    plt.title('Validation accuracy (dashed for test set)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    range_acc = highest_acc - lowest_acc
    plt.ylim([lowest_acc-range_acc*0.1, highest_acc+range_acc*0.1])
    plt.legend()
    fig.savefig(unique_file + '_test_accuracy.svg')
    plt.close(fig)

    export_dictionary = dict(dataset_name=dataset_name,
                             random_state=random_state, beta=beta,
                             m_method=m_method,
                             last_train_index=last_train_index)
    export_dictionary = {**export_dictionary, **test_acc_dict}
    CSV ="\n".join([k+','+str(v) for k,v in export_dictionary.items()])
    #You can store this CSV string variable to file as below
    with open(unique_file + "_summary.csv", "w") as file:
        file.write(CSV)

    generate_summary(dataset_name, output_folder)

if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
