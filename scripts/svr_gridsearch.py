import utility
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

X = utility.load_encoded('train')
y = utility.load_truth()
Xtest = utility.load_encoded('test')

tuned_parameters = {'C': [0.8, 1.0, 1.2],
                    'kernel':['rbf'],
                    'degree':[2, 3, 4],
                    'gamma':[0.0],
                    'probability':[True]}

clf = GridSearchCV(SVR(), tuned_parameters, cv=3,
                   score_func=utility.eval_auc)
clf.fit(X, y)

for params, avgscore, scores in clf.grid_scores_ :
    print avgscore, params

pred = clf.best_estimator_.predict(X)
print pred
