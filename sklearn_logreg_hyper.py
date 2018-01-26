import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV


from experiments.data import load_webs
from experiments.visualizations import plot_errorbar

seed = 42

training, validation, classes = load_webs(random_state=seed)

X_t, Z_t, z_t = training
X_v, Z_v, z_v, Y_v, y_v = validation

def search_1():
    fold = KFold(len(y_v), n_folds=10, shuffle=True, random_state=seed)

    Cs = list(np.power(10.0, np.arange(-10, 10)))

    for penalty in ['l1', 'l2']:
        for scoring in ['roc_auc', 'accuracy']:
            searchCV = LogisticRegressionCV(
                Cs=Cs
                ,penalty=penalty
                ,scoring=scoring
                ,cv=fold
                ,random_state=seed
                ,max_iter=10000
                ,fit_intercept=True
                ,solver='newton-cg'
                ,tol=10
            )
            try:
                searchCV.fit(X_v, y_v)
            except:
                continue

            means = searchCV.scores_[1].mean(axis=0)
            best_index = means.argmax()
            print ('Best C {} = {}, mean {} = {}'.format(penalty,
                                                         Cs[best_index],
                                                         scoring,
                                                         means[best_index]))


            fig = plt.figure()
            ax = plot_errorbar(searchCV.scores_[1], perrorevery=1.0, fig=fig,
                               title='{} 10 folds'.format(scoring))
            ax.set_xscale('log')
            ax.set_xlabel(penalty)
            ax.set_ylabel(scoring)
            fig.savefig('sklearn_logreg_hyper_{}_{}.svg'.format(penalty, scoring))

    print('Accuracy on full set = {}'.format(
            np.mean(searchCV.predict(X_v) == y_v)))


param_grid = [
  {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000],
   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']},
 ]
lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid)
clf.fit(X_v, y_v)
sorted(clf.cv_results_.keys())
results = clf.cv_results_
