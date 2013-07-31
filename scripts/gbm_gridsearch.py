import utility
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

X = utility.load_encoded('train')
y = utility.load_truth()
Xtest = utility.load_encoded('test')

tuned_parameters = {'loss': ['huber'],
                    'alpha':[0.99, 0.999],
                    'learning_rate':[0.1],
                    'n_estimators':[1000],
                    'max_depth':[3, 4, 5]}

clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters,
                   score_func=utility.eval_auc, cv=3)
clf.fit(X.toarray(), y)

for params, avgscore, scores in clf.grid_scores_ :
    print avgscore, params

pred = clf.best_estimator_.predict(X)
print pred
