#!/usr/bin/env python
import itertools
import os
import six
from os import walk
import sys
from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, ranksums

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig_extension = 'svg'


def savefig_and_close(fig, figname, path='', bbox_extra_artists=None):
    filename = os.path.join(path, figname)
    fig.savefig(filename, bbox_extra_artists=bbox_extra_artists,
                bbox_inches='tight')
    fig.clear()
    plt.close(fig)


class MyFloat(float):
    def _remove_leading_zero(self, value, string):
        if 1 > value > -1:
            string = string.replace('0', '', 1)
        return string

    def __str__(self):
        string = super(MyFloat, self).__str__()
        return self._remove_leading_zero(self, string)

    def __format__(self, format_string):
        string = super(MyFloat, self).__format__(format_string)
        return self._remove_leading_zero(self, string)


def plot_df_heatmap(df, normalize=None, title='Heat-map',
                    cmap=plt.cm.Blues, colorbar=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    normalize : 'rows', 'cols' (default=None)
    """
    rows = df.index.values
    columns = df.columns.values
    M = df.values

    if normalize == 'rows':
        M = M.astype('float') / M.sum(axis=1)[:, np.newaxis]
    if normalize == 'cols':
        M = M.astype('float') / M.sum(axis=0)[np.newaxis, :]

    h_size = len(columns)*.5 + 2
    v_size = len(rows)*.5 + 2
    fig = plt.figure(figsize=(h_size, v_size))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, interpolation='nearest', cmap=cmap)
    if colorbar:
        fig.colorbar(im)
    ax.set_title(title)
    column_tick_marks = np.arange(len(columns))
    ax.set_xticks(column_tick_marks)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    row_tick_marks = np.arange(len(rows))
    ax.set_yticks(row_tick_marks)
    ax.set_yticklabels(rows)

    thresh = np.nanmin(M) + ((np.nanmax(M)-np.nanmin(M)) / 2.)
    are_ints = df.dtypes[0] in ['int', 'int32', 'int64']
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        # fontsize is adjusted for different number of digits
        if are_ints:
            ax.text(j, i, M[i, j], horizontalalignment="center",
                    verticalalignment="center", color="white" if M[i, j] >
                    thresh else "black")
        else:
            if np.isfinite(M[i, j]):
                ax.text(j, i, '{:0.2f}'.format(MyFloat(M[i, j])),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if M[i, j] > thresh else "black")

    ax.set_ylabel(df.index.name)
    ax.set_xlabel(df.columns.name)
    fig.tight_layout()
    return fig


def get_list_results_folders(folder, essentials=['description.txt'],
                             finished=None, return_unfinished=False):
    list_results_folders = []
    list_unfinished_folders = []
    for root, subdirs, files in walk(folder):
        if set(essentials).issubset(set(files)):
            if set(finished).issubset(set(files)):
                list_results_folders.append(root)
            elif return_unfinished:
                list_unfinished_folders.append(root)

    if return_unfinished:
        return list_results_folders, list_unfinished_folders

    return list_results_folders


def format_diary_df(df):
    df[2] = pd.to_datetime(df[2])
    df[3] = pd.to_timedelta(df[3], unit='s')

    new_column_names = {0: 'entry_n', 1: 'subentry_n', 2: 'date', 3: 'time'}
    for i in range(5, df.shape[1], 2):
        new_column_names[i] = df.ix[0, i-1]
    df.rename(columns=new_column_names, inplace=True)

    df.drop(list(range(4, df.shape[1], 2)), axis=1, inplace=True)
    return df


def get_dataframe_from_csv(folder, filename, keep_time=False):
    filename = os.path.join(folder, filename)
    df = pd.read_csv(filename, header=None, quotechar='|',
                     infer_datetime_format=True)
    df = format_diary_df(df)
    if keep_time:
        to_drop = ['entry_n', 'subentry_n']
    else:
        to_drop = ['entry_n', 'subentry_n', 'date', 'time']
    df.drop(to_drop, axis=1, inplace=True)
    return df


def extract_summary(folder):
    dataset_df = get_dataframe_from_csv(folder, 'dataset.csv', keep_time=True)
    results_df = get_dataframe_from_csv(folder, 'training.csv',
                                        keep_time=False)

    best_epoch = results_df.groupby(as_index='pid', by='epoch')['val_y_loss'].mean().argmin()
    # FIXME the best epoch could be computed for all the summaries later on
    # However, it seems easier at this point
    results_df['best_epoch'] = best_epoch
    model_df = get_dataframe_from_csv(folder, 'model.csv', keep_time=False)

    dataset_df['folder'] = folder
    results_df['folder'] = folder
    model_df['folder'] = folder

    summary = pd.merge(results_df, dataset_df)
    summary = pd.merge(summary, model_df)

    return summary


def extract_unfinished_summary(folder):
    dataset_df = get_dataframe_from_csv(folder, 'dataset.csv', keep_time=True)

    dataset_df['folder'] = folder

    return dataset_df


def export_datasets_info(df, path='', stylesheets=['style.css']):
    columns = ['dataset', 'n_samples_without_y', 'n_samples_with_y', 'n_features',
               'n_classes']
    sort_by = ['dataset']
    index = columns[0]
    df_table = df[columns].drop_duplicates(subset=columns, keep='first'
                                           ).sort_values(sort_by).set_index(index)
    # Export to LaTeX
    df_table.to_latex(os.path.join(path, "datasets.tex"))

    # Add mean performance of best model
    best_model = df.groupby('architecture')['val_y_acc'].mean().argmax()
    mean_loss_best_model = df[df['architecture'] == best_model][['dataset',
        'val_y_acc']].groupby(by='dataset').mean().round(decimals=2)
    mean_loss_best_model = mean_loss_best_model.rename(columns={'val_y_acc':
        'mean(val_y_acc)'})
    df_table = pd.concat([df_table, mean_loss_best_model], axis=1)

    # FIXME study if this change needs to be done in export_df
    df_table_one_index = df_table.reset_index()
    # Export to svg
    export_df(df_table_one_index, 'datasets', path=path, extension='svg')

    return df_table


def export_df(df, filename, path='', extension='pdf',
              stylesheets=['style.css']):
    filename = os.path.join(path, filename)
    ax = render_mpl_table(df)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig("{}.{}".format(filename, extension))


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'],
                     edge_color='w', bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    source: https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def friedman_test(df, index, column):
    indices = np.sort(df[index].unique())

    results = {}
    first = True
    for ind in indices:
        results[ind] = df[df[index] == ind][column].values
        if first:
            size = results[ind].shape[0]
            first = False
        elif size != results[ind].shape[0]:
            print("Friedman test can not be done with different sample sizes")
            return

    # FIXME be sure that the order is correct
    statistic, pvalue = friedmanchisquare(*results.values())
    return statistic, pvalue


def wilcoxon_rank_sum_test(df, index, column, signed=False,
                           twosided=True):
    indices = np.sort(df[index].unique())
    results = {}
    for i, index1 in enumerate(indices):
        results[index1] = df[df[index] == index1][column]
        # Ensures that all the results are aligned by simulation number (column
        # sim), dataset (column name) and mixing_matrix_M
        if i == 0:
            experiments = df[df[index] == index1][['sim', 'mixing_matrix_M',
                                                   'dataset']].values
        else:
            np.testing.assert_equal(experiments,
                                    df[df[index] == index1][['sim',
                                                             'mixing_matrix_M',
                                                             'dataset']].values)
    stat = []
    for (index1, index2) in itertools.combinations(indices, 2):
        if index1 != index2:
            if signed:
                statistic, pvalue = wilcoxon(results[index1].values,
                                             results[index2].values)
            else:
                statistic, pvalue = ranksums(results[index1].values,
                                             results[index2].values)
            if not twosided:
                pvalue /= 2
            stat.append(pd.DataFrame([[index1, index2, statistic, pvalue]],
                                     columns=['index1', 'index2', 'statistic',
                                              'p-value']))

    dfstat = pd.concat(stat, axis=0, ignore_index=True)
    return dfstat


def main(results_path='results', summary_path='', filter_rows={},
         filter_performance=1.0, verbose=1, gui=False):
    print('\n#########################################################')
    print('##### Making summary of folder {}'.format(results_path))
    print('#')
    results_folders, unfin_folders = get_list_results_folders(
            results_path, essentials=['description.txt', 'dataset.csv',
                                      'model.csv'],
            finished=['validation.csv', 'training.csv'],
            return_unfinished=True)

    if summary_path == '':
        summary_path = os.path.join(results_path, 'summary')

    # Creates summary path if it does not exist
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    u_summaries = []
    for uf in unfin_folders:
        u_summaries.append(extract_unfinished_summary(uf))

    if len(u_summaries) > 0:
        print("Experiments that did not finish: {}".format(len(u_summaries)))
        dfs_unf = pd.concat(u_summaries, axis=0, ignore_index=True)
        if verbose > 1:
            print(dfs_unf)

    f_summaries = []
    for rf in results_folders:
        f_summaries.append(extract_unfinished_summary(rf))

    if len(f_summaries) == 0:
        print("There are no finished experiments")
        sys.exit(0)
    else:
        print("Experiments finished: {}".format(len(f_summaries)))

    dfs_fin = pd.concat(f_summaries, axis=0, ignore_index=True)

    if len(f_summaries) > 0 and verbose > 2:
        print(dfs_fin)

    summaries = []
    for rf in results_folders:
        summaries.append(extract_summary(rf))

    df = pd.concat(summaries, axis=0, ignore_index=True)

    # Remove experiments with no information about the dataset
    df = df[df.dataset.notnull()]

    if len(filter_rows) > 0:
        print("Filtering only rows that contain")
        for key, value in filter_rows.items():
            print("- [{}] = {}".format(key, value))
            if df[key].dtype == 'object':
                df = df[df[key].str.contains(value)]
            else:
                df = df[df[key] == float(value)]

    df['basename'] = df.folder.apply(os.path.basename)
    # Definition of a different experiment setup
    exp_setup_info = ['basename', 'dataset', 'method', 'training_method',
                      'architecture', 'loss', 'init', 'input_dim', 'n_classes',
                      'n_features', 'epochs', 'n_samples_with_y',
                      'n_samples_without_y', 'lr', 'l1', 'l2', 'optimizer',
                      'nesterov', 'decay', 'momentum']
    exp_setup_with_repetitions = list(exp_setup_info).append('pid')

    # Keep only the last computed results for the same experiment
    df.sort_values(by=['date', 'time'], ascending=True, inplace=True)

    df.drop_duplicates(subset=exp_setup_with_repetitions, inplace=True,
                       keep='last')

    ########################################################################
    # Export information about the datasets
    ########################################################################
    export_datasets_info(df, path=summary_path)

    ########################################################################
    # Export information about the experimental setup
    ########################################################################
    experimental_setup = df.groupby(exp_setup_info).size()
    df_exp_setup = experimental_setup.to_frame().reset_index()
    df_exp_setup.to_csv(os.path.join(summary_path, "experimental_setup.csv"),
                        header=True)
    # Export to pdf
    export_df(df_exp_setup, 'experimental_setup', path=summary_path,
              extension='svg')
    if verbose > 0:
        print('The different experimental setups are:')
        print(df_exp_setup)

    ########################################################################
    # Boxplots by different groups
    ########################################################################
    groups_by = ['architecture', 'dataset', 'method']
    columns = ['val_y_acc']
    for groupby in groups_by:
        for column in columns:
            # grouped = df[idx].groupby([groupby])
            grouped = df.groupby([groupby])

            df2 = pd.DataFrame({col: vals[column] for col, vals in grouped})
            meds = df2.median()
            meds.sort_values(ascending=False, inplace=True)
            df2 = df2[meds.index]

            fig = plt.figure(figsize=(10, len(meds)/2+3))
            ax = df2.boxplot(vert=False)
            ax.set_title('results grouped by {}'.format(groupby))

            counts = {k: len(v) for k, v in grouped}
            ax.set_yticklabels(['%s\n$n$=%d' % (k, counts[k]) for k in meds.keys()])
            ax.set_xlabel(column)
            savefig_and_close(fig, '{}_{}.{}'.format(groupby, column,
                                                     fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of models vs architecture
    ########################################################################
    indices = ['architecture']
    columns = ['dataset']
    values = ['val_y_acc']
    for value in values:
        for index in indices:
            for column in columns:
                df2 = pd.pivot_table(df, values=value, index=index,
                                     columns=column,
                                     aggfunc=len)
                fig = plot_df_heatmap(df2, title='Number of experiments',
                                      cmap=plt.cm.Greys)
                savefig_and_close(fig, '{}_vs_{}_{}_heatmap_count.{}'.format(
                            index, column, value, fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of finished experiments dataset vs mixing_matrix_M
    ########################################################################
    df2 = pd.pivot_table(df, values='val_y_acc', index='dataset', columns='architecture',
                         aggfunc=len)
    df2 = df2.fillna(0).astype(int)
    fig = plot_df_heatmap(df2, title='Number of experiments',
                          cmap=plt.cm.Greys)
    savefig_and_close(fig, 'dataset_vs_architecture_heatmap_count.{}'.format(
                        fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of method vs architecture or dataset
    ########################################################################
    # TODO annotate axis
    indices = ['method']
    columns = ['architecture', 'dataset']
    values = ['val_y_acc']
    normalizations = [None, 'rows', 'cols']
    for value in values:
        for index in indices:
            for column in columns:
                df2 = pd.pivot_table(df, values=value, index=index,
                                     columns=column, aggfunc=len, fill_value=0).astype(int)
                fig = plot_df_heatmap(df2, title='Number of experiments',
                                      cmap=plt.cm.Greys)
                savefig_and_close(fig, '{}_vs_{}_{}_heatmap_count.{}'.format(
                            index, column, value, fig_extension), path=summary_path)
                for norm in normalizations:
                    # Mean error rates
                    df2 = pd.pivot_table(df, values=value, index=index,
                                         columns=column,
                                         aggfunc=np.mean)
                    title = r'Mean acc'
                    fig = plot_df_heatmap(df2, normalize=norm, title=title)
                    savefig_and_close(fig, '{}_vs_{}_{}_heatmap_{}.{}'.format(
                                index, column, value, norm, fig_extension), path=summary_path)

                    # Median error rates
                    df2 = pd.pivot_table(df, values=value, index=index,
                                         columns=column,
                                         aggfunc=np.median)
                    title = r'Median acc'
                    fig = plot_df_heatmap(df2, normalize=norm, title=title)
                    savefig_and_close(fig, '{}_vs_{}_{}_median_heatmap_{}.{}'.format(
                                index, column, value, norm, fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of method vs architecture for every dataset
    ########################################################################
    filter_by_column = 'dataset'
    filter_values = df[filter_by_column].unique()
    # TODO change columns and indices
    indices = ['method']
    columns = ['architecture', 'n_samples_with_y', 'n_classes', 'n_features']
    values = ['val_y_acc']
    normalizations = [None, 'rows', 'cols']
    for filtered_row in filter_values:
        for value in values:
            for index in indices:
                for column in columns:
                    df_filtered = df[df[filter_by_column] == filtered_row]
                    df2 = pd.pivot_table(df_filtered, values=value,
                                         index=index, columns=column)
                    if df2.columns.dtype in ['object', 'str']:
                        for norm in normalizations:
                            title = r'Heat-map by {}'.format(filtered_row)
                            fig = plot_df_heatmap(df2, normalize=norm, title=title)
                            savefig_and_close(fig, '{}_vs_{}_by_{}_{}_heatmap_{}.{}'.format(
                                        index, column, filtered_row, value, norm,
                                        fig_extension), path=summary_path)

                        # Export the counting of experiments of the filtered
                        # column and value
                        df3 = pd.pivot_table(df_filtered, values=value, index=index,
                                             columns=column, aggfunc=len,
                                             fill_value=0).astype(int)
                        fig = plot_df_heatmap(df3, title='Number of experiments',
                                              cmap=plt.cm.Greys)
                        savefig_and_close(fig, '{}_vs_{}_by_{}_{}_heatmap_count.{}'.format(
                                    index, column, filtered_row, value, fig_extension), path=summary_path)
                    else:
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        df2.transpose().plot(ax=ax, style='.-', logx=True)
                        ax.set_title('{} {}'.format(filtered_row, value))
                        ax.set_ylabel(value)
                        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, .5))
                        savefig_and_close(fig,
                                          'plot_mean_{}_vs_{}_by_{}_{}.{}'.format(
                                              index, column, filtered_row,
                                              value, fig_extension),
                                          path=summary_path,
                                          bbox_extra_artists=(lgd,))

    ########################################################################
    # Boxplots by Experimental setup and dataset
    ########################################################################
    filter_by_column = 'dataset'
    filter_values = df[filter_by_column].unique()
    groupby = exp_setup_info
    columns = ['val_y_acc']
    for filtered_row in filter_values:
        for column in columns:
            df_filtered = df[df[filter_by_column] == filtered_row]
            grouped = df_filtered.groupby(groupby)

            df2 = pd.DataFrame({col: vals[column] for col, vals in grouped})
            meds = df2.median()
            meds.sort_values(ascending=False, inplace=True)
            df2 = df2[meds.index]

            fig = plt.figure(figsize=(10, len(meds)/2+3))
            ax = df2.boxplot(vert=False)
            ax.set_title('results grouped by experimental setup')

            counts = {k: len(v) for k, v in grouped}
            ax.set_yticklabels(['%s\n$n$=%d' % (k, counts[k]) for k in meds.keys()])
            ax.set_xlabel(column)
            savefig_and_close(fig, '{}_{}_{}.{}'.format(filter_by_column,
                filtered_row, column, fig_extension), path=summary_path)

    if gui:
        import dfgui
        dfgui.show(df)


def __test_1():
    main('../results', summary_path='summary')


def parse_arguments():
    parser = ArgumentParser(description=("Generates a summary of all the " +
                                         "experiments in the subfolders of " +
                                         "the specified path"))
    parser.add_argument("results_path", metavar='PATH', type=str,
                        default='results',
                        help="Path with the result folders to summarize.")
    parser.add_argument("summary_path", metavar='SUMMARY', type=str,
                        default='', nargs='?',
                        help="Path to store the summary.")
    parser.add_argument("-p", "--performance", metavar='PERFORMANCE',
                        type=float, default=1.0, dest='filter_performance',
                        help="Filter all the datasets with supervised performance lower than the specified value")
    parser.add_argument("-f", "--filter", type=json.loads,
                        default='{}', dest='filter_rows',
                        help="Dictionary with columns and filters.")
    parser.add_argument("-v", "--verbose", type=int,
                        default=1, dest='verbose',
                        help="Dictionary with columns and filters.")
    parser.add_argument("-g", "--gui", default=False,
                        action='store_true', dest='gui',
                        help="Open a GUI with the dataframe.")
    return parser.parse_args()


if __name__ == '__main__':
    # __test_1()
    args = parse_arguments()
    main(**vars(args))
