
# coding: utf-8

# In[1]:


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

import sys

if is_interactive():
    get_ipython().magic(u'matplotlib inline')
    sys.path.append('../')

import numpy
from experiments.data import make_weak_true_partition
from wlc.WLweakener import computeM, weak_to_index, estimate_M
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from experiments.models import create_model, MyKerasClassifier
import inspect
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import cm

from experiments.visualizations import plot_heatmap
from experiments.visualizations import plot_confusion_matrix

plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams["figure.dpi"] = 100

cmap = cm.get_cmap('Accent')

from cycler import cycler
default_cycler = (cycler(color=['darkred', 'forestgreen', 'darkblue', 'violet', 'darkorange', 'saddlebrown']) +
                  cycler(linestyle=['-', '--', '-.', '-', '--', '-.']) +
                  cycler(marker=['o', 'v', 'x', '*', '+', '.']) +
                  cycler(lw=[2, 1.8, 1.6, 1.4, 1.2, 1]))

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=default_cycler)

# ## 5.b. Load saved results

# In[2]:


import os
import re
import pandas

if is_interactive():
    path = '../results_mnist_1959/'
else:
    path = sys.argv[1]

article_version = True
if article_version:
    plt.rcParams['figure.figsize'] = (3, 2)
    plt.rcParams["figure.dpi"] = 300


def load_all_json(results_path, expression=".*.json"):
    regexp = re.compile(expression)
    filename_list = []
    df_list = []
    for root, subdirs, files in os.walk(results_path, followlinks=True):
        file_list = list(filter(regexp.match, files))
        for filename in file_list:
            if filename in filename_list:
                continue
            filename_list += filename
            try:
                df_list.append(pandas.read_json(os.path.join(root, filename)).T)
                df_list[-1]['filename'] = filename
            except pandas.errors.EmptyDataError as e:
                print(e)
                print('Classifier = {}, filename = {}'.format(classifier,
                    filename))

    if df_list:
        df = pandas.concat(df_list)
    else:
        df = pandas.DataFrame()
    return df

df_exp_all = load_all_json(path, expression=".*.json")

print(df_exp_all.columns)
dataset_list = df_exp_all['dataset_name'].unique()

for dataset_name in dataset_list:
    df_experiment_all = df_exp_all[df_exp_all['dataset_name'] == dataset_name]

    #locals().update(df_experiment)
    print(df_experiment_all)
    print(df_experiment_all.columns)


    # # Filter results for only one dataset

    # In[3]:


    # Case with multiple noise methods and levels in a grid
    if 'M_method' in df_experiment_all.columns:
        M_method_list = df_experiment_all['M_method'].unique()
        M_beta_list = df_experiment_all['M_beta'].unique()
        print(M_method_list)
        print(M_beta_list)

        for M_method in M_method_list:
            for M_beta in M_beta_list:
                df_experiment = df_experiment_all[(df_experiment_all['M_method'] == M_method) &
                                                  (df_experiment_all['M_beta'] == M_beta)].copy()
                print(df_experiment)
                if len(df_experiment) == 0:
                    continue

                acc_lowerbound = df_experiment['acc_lowerbound'].mean()
                acc_upperbound = df_experiment['acc_upperbound'].mean()
                M_alpha = df_experiment['M_alpha'].mean()
                n_wt_samples = df_experiment['n_wt_samples'].max()
                n_wt_samples_train = df_experiment['n_wt_samples_train'].max()
                n_weak_labels = df_experiment['n_samples'].max()
                n_wt_samples_val = df_experiment['n_wt_samples_val'].max()
                n_wt_samples_test = df_experiment['n_wt_samples_test'].max()
                n_iterations = df_experiment.shape[0]
                weak_proportions = df_experiment['weak_proportions'].iloc[0]

                acc_methods_list = df_experiment['acc_methods'].values

                acc = {}
                for method in acc_methods_list[0].keys():
                    acc[method] = numpy.mean([row[method] for row in acc_methods_list], axis=0)

                print(n_weak_labels)
                print(weak_proportions)

                print(acc['Weak'])

                if M_method is not None:
                    M_text = '_{}_a{:02.0f}_b{:02.0f}'.format(M_method, 10*M_alpha, 10*M_beta)
                else:
                    M_text = ''
                filename = 'full_vs_em_{}_{}{}'.format(dataset_name, n_wt_samples_train, M_text)

                if acc_upperbound is not None:
                    print('Acc. Upperbound = {}'.format(acc_upperbound))
                for key, value in sorted(acc.items()):
                    print('Acc. {}\n{}'.format(key, value))
                print('Acc. Lowerbound = {}'.format(acc_lowerbound))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                if M_method is not None:
                    M_text = "\n" + r"{} $\alpha={:0.1f}$, $\beta={:0.1f}$".format(M_method, M_alpha, M_beta)
                else:
                    M_text = ''
                if not article_version:
                    ax.set_title("All methods used {} train. and {} valid. true labels.{}".format(
                                n_wt_samples_train, n_wt_samples_val, M_text))
                for key, value in sorted(acc.items()):
                    ax.plot(weak_proportions, value, label='{}'.format(key, n_wt_samples_train))
                if acc_upperbound is not None:
                    ax.axhline(y=acc_upperbound, color='black', lw=2,linestyle='--', label='Superv. (+{} true)'.format(n_wt_samples))
                ax.axhline(y=acc_lowerbound, color='gray', lw=2, linestyle='-.', label='Supervised')
                #ax.set_ylim([0, 1])
                ax.set_xlabel('# weak samples')
                ax.set_ylabel('Mean Acc. (#it={})'.format(n_iterations))
                ax.set_xscale("symlog")
                if not article_version:
                    ax.legend(loc=0, fancybox=True, framealpha=0.8)
                ax.grid()
                fig.tight_layout()
                fig.savefig(filename + '.svg')
                print('Saved figure as {}.svg'.format(filename))

        if article_version:
            import pylab
            # create a second figure for the legend
            figLegend = pylab.figure()

            # produce a legend for the objects in the other figure
            pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'center')
            figLegend.tight_layout()
            figLegend.savefig(filename + '_legend.svg')


    # In[4]:


    n_wt_samples_train_list = df_experiment_all['n_wt_samples_train'].unique()

    if 'M_method_list' in df_experiment_all.columns:
        M_method_list = df_experiment_all['M_method_list'].iloc[0]
        if (len(M_method_list) == 1) and (M_method_list[0] == None):
            for n_wt_samples_train in n_wt_samples_train_list:
                df_experiment = df_experiment_all[(df_experiment_all['n_wt_samples_train'] == n_wt_samples_train)].copy()

                print(df_experiment)
                if len(df_experiment) == 0:
                    continue

                print(df_experiment)

                acc_lowerbound = df_experiment['acc_lowerbound'].mean()
                acc_upperbound = df_experiment['acc_upperbound'].mean()
                M_alpha = df_experiment['M_alpha'].mean()
                n_wt_samples = df_experiment['n_wt_samples'].max()
                n_wt_samples_train = df_experiment['n_wt_samples_train'].max()
                n_weak_labels = df_experiment['n_samples'].max()
                n_wt_samples_val = df_experiment['n_wt_samples_val'].max()
                n_wt_samples_test = df_experiment['n_wt_samples_test'].max()
                n_iterations = df_experiment.shape[0]
                weak_proportions = df_experiment['weak_proportions'].iloc[0]


                weak_proportions = df_experiment['weak_proportions'].iloc[0]

                acc_methods_list = df_experiment['acc_methods'].values

                acc = {}
                for method in acc_methods_list[0].keys():
                    acc[method] = numpy.mean([row[method] for row in acc_methods_list], axis=0)

                print(n_weak_labels)
                print(weak_proportions)

                print(acc['Weak'])

                if M_method is not None:
                    M_text = '_{}_a{:02.0f}_b{:02.0f}'.format(M_method, 10*M_alpha, 10*M_beta)
                else:
                    M_text = ''
                filename = 'full_vs_em_{}_{}{}'.format(dataset_name, n_wt_samples_train, M_text)

                print(acc_upperbound)
                if numpy.isfinite(acc_upperbound):
                    print('Acc. Upperbound = {}'.format(acc_upperbound))
                for key, value in sorted(acc.items()):
                    print('Acc. {}\n{}'.format(key, value))
                print('Acc. Lowerbound = {}'.format(acc_lowerbound))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                if M_method is not None:
                    M_text = "\n" + r"{} $\alpha={:0.1f}$, $\beta={:0.1f}$".format(M_method, M_alpha, M_beta)
                else:
                    M_text = ''
                if not article_version:
                    ax.set_title("All methods used {} train. and {} valid. true labels.{}".format(
                                n_wt_samples_train, n_wt_samples_val, M_text))
                for key, value in sorted(acc.items()):
                    ax.plot(weak_proportions, value, label='{}'.format(key, n_wt_samples_train))
                if numpy.isfinite(acc_upperbound):
                    ax.axhline(y=acc_upperbound, color='black', lw=2,linestyle='--', label='Superv. (+{} true)'.format(n_wt_samples))
                ax.axhline(y=acc_lowerbound, color='gray', lw=2, linestyle='-.', label='Supervised')
                ax.set_xlabel('# weak samples')
                ax.set_ylabel('Mean Acc. (#it={})'.format(n_iterations))
                ax.set_ylim([0, 1])
                #ax.set_xscale("symlog")
                if not article_version:
                    ax.legend(loc=0, fancybox=True, framealpha=0.8)
                ax.grid()
                fig.tight_layout()
                fig.savefig(filename + '.svg')
                print('Saved figure as {}.svg'.format(filename))

        if article_version:
            import pylab
            # create a second figure for the legend
            figLegend = pylab.figure()

            # produce a legend for the objects in the other figure
            pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'center')
            figLegend.tight_layout()
            figLegend.savefig(filename + '_legend.svg')


    # In[5]:


    # Case with multiple noises combined together
    if 'M_method_list' in df_experiment_all.columns:
        M_method = df_experiment_all['M_method_list'].iloc[0]
        M_beta_list = df_experiment_all['M_beta'].unique()
        print(M_method)
        print(M_beta_list)

        # For the multiple datasets
        for M_beta in M_beta_list:
            df_experiment = df_experiment_all[(df_experiment_all['M_beta'] == M_beta)].copy()
            print(df_experiment)
            if len(df_experiment) == 0:
                continue
            print(M_method)
            acc_upperbound = df_experiment['acc_upperbound'].mean()
            M_alpha = df_experiment['M_alpha'].mean()
            n_weak_labels = df_experiment['n_samples'].max()
            n_wt_samples_train = df_experiment['n_wt_samples_train'].max()
            n_wt_samples_val = df_experiment['n_wt_samples_val'].max()
            n_iterations = df_experiment.shape[0]
            weak_proportions = df_experiment['weak_proportions'].iloc[0]


            weak_proportions = df_experiment['weak_proportions'].iloc[0]

            acc_methods_list = df_experiment['acc_methods'].values

            n_iterations = df_experiment.shape[0]
            print(M_method)

            acc_methods_list = df_experiment['acc_methods'].values

            acc = {}
            for method in acc_methods_list[0].keys():
                acc[method] = numpy.mean([row[method] for row in acc_methods_list], axis=0)

            print(acc['Weak'])

            if M_method is not None:
                M_text = '_{}_a{:02.0f}_b{:02.0f}'.format(M_method, 10*M_alpha, 10*M_beta)
            else:
                M_text = ''
            filename = 'full_vs_em_{}_{}{}'.format(dataset_name, n_wt_samples_train, M_text)

            if acc_upperbound is not None:
                print('Acc. Upperbound = {}'.format(acc_upperbound))
            for key, value in sorted(acc.items()):
                print('Acc. {}\n{}'.format(key, value))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            if M_method is not None:
                M_text = "\n" + r"{} $\alpha={:0.1f}$, $\beta={:0.1f}$".format(M_method, M_alpha, M_beta)
            else:
                M_text = ''
            if not article_version:
                ax.set_title("All methods used {} train. and {} valid. true labels.{}".format(
                            n_wt_samples_train, n_wt_samples_val, M_text))
            for key, value in sorted(acc.items()):
                ax.plot(weak_proportions, value, label='{}'.format(key, n_wt_samples_train))
            if acc_upperbound is not None:
                ax.axhline(y=acc_upperbound, color='black', lw=2,linestyle='--', label='Superv. (+{} true)'.format(n_wt_samples_train))
            ax.set_xlabel('# weak samples')
            ax.set_ylabel('Mean Acc. (#it={})'.format(n_iterations))
            ax.set_xscale("symlog")
            if not article_version:
                ax.legend(loc=0, fancybox=True, framealpha=0.8)
            ax.grid()
            fig.tight_layout()
            fig.savefig(filename + '.svg')
            print('Saved figure as {}.svg'.format(filename))

        if article_version:
            import pylab
            # create a second figure for the legend
            figLegend = pylab.figure()

            # produce a legend for the objects in the other figure
            pylab.figlegend(*ax.get_legend_handles_labels(), loc = 'center')
            figLegend.tight_layout()
            figLegend.savefig(filename + '_legend.svg')


    # In[6]:


    print(df_experiment_all.columns)
    df_experiment_all['acc_upperbound'] = pandas.to_numeric(df_experiment_all['acc_upperbound'])
    print(df_experiment_all['acc_upperbound'].mean())
    print(df_experiment_all['acc_upperbound'].min())
    print(df_experiment_all['acc_upperbound'].max())
    print(df_experiment_all.dtypes)

    print(df_experiment_all['M_beta'].unique())
    print(df_experiment_all['M_alpha'].unique())


    # In[7]:


    df_experiment_all.pivot_table(values='acc_upperbound', index='M_beta', aggfunc='mean')

