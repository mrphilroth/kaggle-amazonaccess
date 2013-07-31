import utility
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

dftest = pd.io.parsers.read_csv('weakgbms_test_predictions.csv')
dftrain = pd.io.parsers.read_csv('weakgbms_train_predictions.csv')

y = np.array(dftrain.ACTION)
del dftrain['id']
del dftrain['ACTION']
X = np.array(dftrain)

Xtest = np.array(dftest)[:,1:]

lgr = LogisticRegression(penalty='l2', C=1.0)
lgr.fit(X, y)
print utility.roc_curve(y, np.clip(lgr.predict(X), 0, 1))

dftest['ACTION'] = pd.Series(np.clip(lgr.predict(Xtest), 0, 1))
utility.write_submission(dftest, 'weakgbms_ensemble_lgr_18.csv')
