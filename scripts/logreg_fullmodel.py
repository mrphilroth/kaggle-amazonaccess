import os
import pickle
import utility
import numpy as np
import pandas as pd
from scipy import sparse, optimize
from sklearn import metrics, cross_validation, linear_model

def fopt_pred(pars, data) :
    pars = pars / pars.sum()
    pred = np.dot(data, pars)
    return np.reshape(pred, (len(pred), 1))

def fopt(pars, truth, data) :
    auc = -metrics.auc_score(truth, fopt_pred(pars, data))
    return auc

def model_generate_level1_test(bestc, features, X_train, X_test, y) :

    ntrain = X_train.shape[0]

    Xt = np.vstack((X_train[:,features], X_test[:,features]))
    Xt, keymap = utility.OneHotEncoder(Xt)
    X_train = Xt[:ntrain]
    X_test = Xt[ntrain:]

    model = linear_model.LogisticRegression()
    model.C = bestc

    model.fit(X_train, y)

    return model.predict_proba(X_test)[:,1]

def generate_level1_test(mlist, X_train, X_test, y) :

    X_level1_test = None
    for modelstr in mlist :
        print "Training {}...".format(modelstr)

        featfn = '{}/{}.dat'.format(utility.ddir, modelstr)
        paramfn = featfn.replace('.dat', '_bestc.txt')

        features = np.load(featfn)
        bestc = float(open(paramfn, 'r').read().strip())

        newdata = model_generate_level1_test(
            bestc, features, X_train, X_test, y)

        if X_level1_test == None :
            X_level1_test = newdata
        else :
            X_level1_test = np.vstack((X_level1_test, newdata))

    return np.transpose(X_level1_test)

def main() :

    print "Loading data..."
    X_train = utility.load_encoded('train')
    X_test = utility.load_encoded('test')
    y = utility.load_truth()

    mlist = []
    for featfn in os.listdir(utility.ddir) :
        if not (featfn.startswith('feateng') and
                featfn.endswith('dat')) : continue
        modelstr = os.path.splitext(featfn)[0]
        featfn = utility.ddir + '/' + featfn
        if not os.path.exists(featfn) : continue
        paramfn = featfn.replace('.dat', '_bestc.txt')
        if not os.path.exists(paramfn) : continue

        mlist.append(modelstr)

    print "Generating level1 test data..."
    X_level1_test = None
    X_level1_testfn = utility.ddir + '/fullmodel_precombined.dat'
    if os.path.exists(X_level1_testfn) :
        X_level1_test = np.load(X_level1_testfn)
    else :
        X_level1_test = generate_level1_test(mlist, X_train, X_test, y)
        X_level1_test.dump(X_level1_testfn)

    print "Writing submissions..."
    weightfn = 'logreg_level1weights_rev{}.dat'.format(utility.logregrev)
    weights = np.load(utility.ddir + '/' + weightfn)
    final_submission = fopt_pred(weights, X_level1_test)
    utility.create_test_submission(
        'logreg_stacked_preds_rev{}.csv'.format(utility.logregrev),
        np.ravel(final_submission))

    print "Getting gbm trained models..."
    gbrfn = '{}/gbr_nest1000.csv'.format(utility.subdir)
    gbmone = np.array(pd.io.parsers.read_csv(gbrfn)['Action'])
    x_level1_test = np.transpose(np.vstack((X_level1_test.T, gbmone)))

    lgwfn = 'logreg_level1weights_linreg_rev{}.dat'.format(utility.logregrev)
    lgweights = np.load(utility.ddir + '/' + lgwfn)
    final_linregsubmission = fopt_pred(lgweights, X_level1_test)
    utility.create_test_submission(
        'logreg_stacked_preds_linreg_rev{}.csv'.format(utility.logregrev),
        np.ravel(final_linregsubmission))

    avgweights = np.ones(len(lgweights)) / float(len(lgweights))
    final_avgsubmission = fopt_pred(avgweights, X_level1_test)
    utility.create_test_submission(
        'logreg_stacked_preds_avg_rev{}.csv'.format(utility.logregrev),
        np.ravel(final_avgsubmission))
    
if __name__ == "__main__":
    main()
