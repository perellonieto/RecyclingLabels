import numpy as np
from scipy.stats import friedmanchisquare

def binarize_weak_labels(z, c):
    """
    Binarizes the weak labels depending on the method used to generate the weak
    labels.

    Args:
        z       :List of weak labels. Each weak label is an integer whose
                 binary representation encondes the observed weak labels
        c       :Number of classes. All components of z must be smaller than
                 2**c
    Returns:
        z_bin
    """
    # Transform the weak label indices in z into binary label vectors
    z_bin = np.zeros((z.size, c), dtype=int)       # weak labels (binary)
    for index, i in enumerate(z):         # From dec to bin
        z_bin[index, :] = [int(x) for x in np.binary_repr(i, width=c)]

    return z_bin


# Merge dictionaries compatible with Python 2 and 3
# source: https://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def compute_friedmanchisquare(table):
    '''
    Arguments:
    ----------
        table: pandas DataFrame of size (n_samples, n_methods)
    Example:
        - n wine judges each rate k different wines. Are any of the k wines
        ranked consistently higher or lower than the others?
    Our Calibration case:
        - n datasets each rate k different calibration methods. Are any of the
        k calibration methods ranked consistently higher or lower than the
        others?
    This will output a statistic and a p-value
    SciPy does the following:
        - k: is the number of parameters passed to the function
        - n: is the lenght of each array passed to the function
    The two options for the given table are:
        - k is the datasets: table['mean'].values).tolist()
        - k is the calibration methods: table['mean'].T.values).tolist()
    '''
    return friedmanchisquare(*(table.T.values).tolist())


# TODO MPN is currently using this one to generate the exported files,
# we need to see the difference between this one and the function
# table_to_latex that outputs on the stout
# TODO should we make the replacements of underscores out of this function for
# the column and row names?
def rankings_to_latex(datasets, table, max_is_better=True, scale=1, precision=3,
             table_size="normalsize", caption="", label='table',
             add_std=True, position='tph', column_names=None,
             avg_ranks=None):
    if column_names is None:
        column_names = table.columns
    n_columns = len(column_names)
    row_names = table.index
    n_rows = len(row_names)

    means = table.as_matrix()[:, :n_columns].copy()*scale
    computed_avg_ranks = np.zeros(n_columns)
    stds = table.as_matrix()[:, n_columns:]*scale
    str_table = ("\\begin{table}[" + position + "]\n" +
                 "\\" + table_size + "\n" +
                 "\\centering\n")
    str_columns = "l"
    str_header = ""

    for c_name in column_names:
        str_columns += "c"
        str_header += " & " + c_name.replace('_', r'\_')
    str_header += "\\\\\n"

    str_table += ("\\begin{tabular}{"+str_columns+"}\n" +
                  "\\toprule\n" +
                  str_header +
                  "\\midrule\n")
    for i, name in enumerate(row_names):
        name = name[:10] if len(name) > 10 else name
        name = name.replace("_", r"\_")
        str_row_means = name
        v = means[i]
        v_std = stds[i]
        indices = rankdata(v)
        if max_is_better:
            indices = n_columns + 1 - indices
        for j in np.arange(len(v)):
            idx = indices[j]
            computed_avg_ranks[j] += idx / n_rows
            if idx == 1:
                str_row_means += (" & $\\mathbf{{{0:.{1}f}".format(
                                    v[j], precision))
                if add_std:
                    str_row_means += ("\\pm{0:.{1}f}".format( v_std[j],
                                                             precision))
                str_row_means += ("_{{{0}}}}}$".format(1))

            else:
                idx_s = "{}".format(idx)
                if ".0" in idx_s:
                    idx_s = "{}".format(int(idx))
                str_row_means += (" & ${0:.{1}f}".format( v[j], precision))
                if add_std:
                    str_row_means += ("\\pm{0:.{1}f}".format( v_std[j],
                                                             precision))

                str_row_means += ("_{{{0}}}$".format(idx_s))
        str_table += str_row_means + "\\\\\n"
    str_table += "\\midrule\n"
    str_avg = "avg rank"
    if avg_ranks is None:
        avg_ranks = computed_avg_ranks
    for i in np.arange(n_columns):
        if avg_ranks[i] == min(avg_ranks):
            str_avg += " & \\bf{{{0:.2f}}}".format(avg_ranks[i])
        else:
            str_avg += " & {0:.2f}".format(avg_ranks[i])

    str_table += (str_avg + "\\\\\n" +
                  "\\bottomrule\n" +
                  "\\end{tabular}\n" +
                  "\\normalsize\n" +
                  "\\caption{"+caption+"}\n" +
                  "\\label{"+label+"}\n" +
                  "\\end{table}\n")
    return str_table

