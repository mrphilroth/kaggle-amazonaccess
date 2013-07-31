import utility

import numpy as np
import pandas as pd
import multiprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

X = utility.load_encoded('train')
y = utility.load_truth()
Xtest = utility.load_encoded('test')

def cross_validate(i) :
    X_cvtrain, X_cvtest, y_cvtrain, y_cvtest = train_test_split(
        X, y, test_size=0.2, random_state=i)

    lgr = LogisticRegression(C=2)
    lgr.fit(X_cvtrain, y_cvtrain)
    return i, utility.eval_auc(y_cvtest, lgr.predict_proba(X_cvtest)[:,1])

ncvs = 5
pool = multiprocessing.Pool(5)
res = np.zeros(ncvs)
for i, auc in pool.imap(cross_validate, range(5)) :
    print "{}: {}".format(i, auc)
    res[i] = auc

print "Mean: {}".format(res.mean())

# testgrbpred = gbr.predict(Xtest)
# dftest['ACTION'] = pd.Series(testgrbpred)
# utility.write_submission(dftest, 'gbr_huber_nest1000.csv')

