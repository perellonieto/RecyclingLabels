import os
import errno
import numpy as np
import itertools
import six

from scipy import sparse

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Wedge

from textwrap import wrap


def savefig_and_close(fig, figname, path='', bbox_extra_artists=None):
    filename = os.path.join(path, figname)
    fig.savefig(filename, bbox_extra_artists=bbox_extra_artists,
                bbox_inches='tight')
    fig.clear()
    plt.close(fig)


def newfig(name, **kwargs):
    fig = plt.figure(name, **kwargs)
    fig.clf()
    return fig


def savefig(fig, path='figures', prefix='weak_labels_', extension='svg'):
    fig.tight_layout()
    name = fig.get_label()
    filename = "{}{}.{}".format(prefix, name, extension)
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    fig.savefig(os.path.join(path, filename))


def plot_data(x, y, loc='best', save=True, title='data', cmap='Paired'):
    if sparse.issparse(x):
        x = x.toarray()

    fig = newfig('data', figsize=(5, 3))
    ax = fig.add_subplot(111)

    classes = np.unique(y)
    n_c = float(len(classes))

    cmap = cm.get_cmap(cmap)
    for i, y_i in enumerate(classes):

        ax.scatter(x[(y == y_i).flatten(), 0], x[(y == y_i).flatten(), 1],
                   c=cmap(i/n_c), s=30, edgecolors=None, alpha=.8, lw=0.1,
                   label='{}'.format(y_i))
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_title(title)
    ax.axis('equal')
    ax.legend(loc=loc)
    ax.grid(True)
    if save:
        savefig(fig)
    return fig


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


# TODO use this or other heatmap to visualize confusion matrix
def plot_df_heatmap(df, normalize=None, title='Heat-map',
                    cmap=plt.cm.Blues, colorbar=False, ylabel=None,
                    xlabel=None):
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

    xlabel = df.columns.name
    ylabel = df.index.name

    return plot_heatmap(M, columns=columns, rows=rows, cmap=cmap,
                        colorbar=colorbar, title=title, ylabel=ylabel,
                        xlabel=xlabel)

def plot_confusion_matrix(M, columns=None, rows=None, cmap=plt.cm.Blues,
                          colorbar=False, fig=None, title='Heat-map',
                          ylabel='True label', xlabel='Predicted label'):
    return plot_heatmap(M=M, columns=columns, rows=rows, cmap=cmap,
                        colorbar=colorbar, fig=fig, title=title, ylabel=ylabel,
                        xlabel=xlabel)

def plot_heatmap(M, columns=None, rows=None, cmap=plt.cm.Blues, colorbar=False,
                 fig=None, title='Heat-map', ylabel=None, xlabel=None):
    if columns is None:
        columns = [str(i) for i in range(M.shape[1])]
    if rows is None:
        rows = [str(i) for i in range(M.shape[0])]

    h_size = len(columns)*.5 + 2
    v_size = len(rows)*.5 + 2

    if fig is None:
        fig = plt.figure(figsize=(h_size, v_size))

    ax = fig.add_subplot(111)
    im = ax.imshow(M, interpolation='nearest', cmap=cmap)
    if colorbar:
        fig.colorbar(im)
    if h_size < 4:
        title = "\n".join(wrap(title, 30))
    ax.set_title(title)
    column_tick_marks = np.arange(len(columns))
    ax.set_xticks(column_tick_marks)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    row_tick_marks = np.arange(len(rows))
    ax.set_yticks(row_tick_marks)
    ax.set_yticklabels(rows)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    thresh = np.nanmin(M) + ((np.nanmax(M)-np.nanmin(M)) / 2.)
    are_ints = M.dtype in ['int', 'int32', 'int64']
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        # fontsize is adjusted for different number of digits
        if are_ints:
            num_text = str(M[i, j])
        else:
            num_text = '{:0.2f}'.format(MyFloat(M[i, j]))

        if np.isfinite(M[i, j]):
            ax.text(j, i, num_text, horizontalalignment="center",
                    verticalalignment="center", color="white" if M[i, j] >
                    thresh else "black", fontsize=16-len(num_text))

    fig.tight_layout()
    return fig

def dual_half_circle(center, radius, angle=0, ax=None, colors=('w','k'),
                     **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        #ax.add_artist(wedge)
        ax.add_patch(wedge)
    return [w1, w2]

def test_dual_half_circle_main():
    fig, ax = plt.subplots()
    dual_half_circle((0.5, 0.5), radius=0.3, angle=90, ax=ax)
    ax.axis('equal')
    plt.show()

def plot_multilabel_scatter(X, Y, cmap=cm.get_cmap('Accent'), edgecolor='k',
                            linewidth=0.4, title=None, fig=None,
                            radius_scaler=100.0, **kwargs):
    X_std = X.std(axis=0)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    n_classes = Y.shape[1]

    radius = (X.max() - X.min())/radius_scaler

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.add_subplot(111)
    for x, y in zip(X, Y):
        theta2s = np.cumsum(np.true_divide(y, y.sum())*360.0)
        theta1 = 0
        if np.isfinite(theta2s[0]):
            for i, theta2 in enumerate(theta2s):
                if theta1 != theta2:
                    w = Wedge(x[:2], radius, theta1, theta2, ec=edgecolor, lw=linewidth,
                              fc=cmap(np.true_divide(i, n_classes)), **kwargs)
                    ax.add_patch(w)
                    theta1 = theta2
        else:
            # Not belong to any class
            w = Wedge(x[:2], radius, 0, 360, ec='black', lw=linewidth,
                      fc='white', **kwargs)
            ax.add_patch(w)

    ax.set_xlim([X_min[0]-X_std[0], X_max[0]+X_std[0]])
    ax.set_ylim([X_min[1]-X_std[1], X_max[1]+X_std[1]])
    ax.axis('equal')
    if title is not None:
        ax.set_title(title)
    return fig


def test_multilabel_plot():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 1, 1]])
    plot_multilabel_scatter(X, Y)
    plt.show()


def plot_errorbar(data, x=None, fmt='--o', title='errorbar', elinewidth=1.0,
                  perrorevery=0.2, legend=None, fig=None, **kwargs):
    """

    paramters
        data: np.array or list of np.array
            If it is a list, each np.array is considered as an errorbar line

        errorevery: float
            Percentage of errorbars with respect to the number of samples
    """
    return_fig = fig is None

    if fig is None:
        fig = newfig(title, figsize=(5, 3))

    ax = fig.add_subplot(111)
    ax.set_title(title)

    if type(data) is np.ndarray:
        data = (data,)

    for i, matrix in enumerate(data):
        errorevery = int(1 / perrorevery)
        if errorevery < 1:
            errorevery = 1

        if x is None:
            x = range(matrix.shape[1])

        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)

        ax.errorbar(x=x, y=means, yerr=stds, elinewidth=(len(data)-i)*elinewidth,
                    errorevery=errorevery, capthick=(len(data)-i)*elinewidth,
                    capsize=4, **kwargs)
    if legend is not None:
        ax.legend(legend)

    if return_fig:
        return fig
    return ax


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


def export_df(df, filename, path='', extension='pdf',
              stylesheets=['style.css']):
    """
    df : pandas.DataFrame
    """
    filename = os.path.join(path, filename)
    ax = render_mpl_table(df)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig("{}.{}".format(filename, extension))
