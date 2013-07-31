import utility
import numpy as np
import pandas as pd

scores = []
subs = ['classic', 'smallnum', 'headstart0', 'headstart1']
for sub in subs :
    subfn = utility.subdir + '/logistic_regression_pred_{}.csv'.format(sub)
    arr = np.array(pd.io.parsers.read_csv(subfn)['ACTION'])
    scores.append(arr)

scores = np.mean(np.array(scores), axis=0)
utility.create_test_submission('/last_ditch_effort.csv', scores)
