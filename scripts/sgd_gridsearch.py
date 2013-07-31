import utility
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression

X = utility.load_encoded('train')
y = utility.load_truth()
Xtest = utility.load_encoded('test')

tuned_parameters = {'loss': ['huber'], 'penalty':['l1'],
                    'alpha':[1e-8], 'n_iter':[1000], 'p':[0.1]}
clf = GridSearchCV(SGDRegressor(verbose=1), tuned_parameters,
                   score_func=utility.eval_auc, cv=3)
clf.fit(X, y)

for params, avgscore, scores in clf.grid_scores_ :
    print avgscore, params

pred = clf.best_estimator_.predict(X)
print pred
