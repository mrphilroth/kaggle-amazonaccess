import utility
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

nweakgbms = 18

dftest = utility.load_data('test')
dftrain = utility.load_data('train')

dftestpreds = pd.DataFrame(dftest.id)
dftrainpreds = pd.DataFrame({'id':np.arange(len(dftrain)),
                             'ACTION':dftrain.ACTION})

y = np.array(dftrain.ACTION)
del dftrain['ACTION']
X = np.array(dftrain)

Xtest = np.array(dftest)[:,1:]

def train_weakgbm(i) :

    cols = np.ones(9)
    cols[i % X.shape[1]] = 0
    smallX = np.compress(cols, X, axis=1)

    X_cvtrain, X_cvtest, y_cvtrain, y_cvtest = train_test_split(
        X, y, test_size=0.5, random_state=i)

    gbr = GradientBoostingRegressor(loss='ls', learning_rate=0.1,
                                    n_estimators=1000, max_depth=4,
                                    verbose=1)
    gbr.fit(X_cvtrain, y_cvtrain)

    pred = np.clip(gbr.predict(X_cvtest), 0, 1)
    print i, utility.roc_curve(y_cvtest, pred)

    trainpred = np.clip(gbr.predict(X), 0, 1) 
    testpred = np.clip(gbr.predict(Xtest), 0, 1) 

    return i, trainpred, testpred

pool = multiprocessing.Pool(3)
for i, trainpred, testpred in pool.imap(train_weakgbm, range(nweakgbms)) :
    dftrainpreds['pred{}'.format(i)] = trainpred
    dftestpreds['pred{}'.format(i)] = testpred
    
dftestpreds.to_csv('weakgbms_test_predictions.csv', index=False)
dftrainpreds.to_csv('weakgbms_train_predictions.csv', index=False)
