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

def fnormed(pars, truth, data) :
    toret = 1.0 - np.sum(pars)
    if np.abs(toret) < 1.0e-8 : toret = 0.0
    return toret

def train_level1(X_train, y) :

    ntrain = X_train.shape[0]
    nmodels = X_train.shape[1]

    xopt = None
    iteration = 0
    score_hist = []
    current_models = range(nmodels)
    while len(current_models) > 2 :

        Xt = X_train[:,current_models]
        x0 = np.ones(len(current_models)) / float(len(current_models))
        xbounds = [(0, 1) for x in x0]
        xopt = optimize.fmin_slsqp(fopt, x0, bounds=xbounds,
                                   epsilon=1.0e-1, acc=1e-10,
                                   eqcons=[fnormed], iprint=0,
                                   args=(y, Xt))
        score = -fopt(xopt, y, Xt)
        print "Iteration {} Mean AUC: {}".format(iteration, score)
        score_hist.append((score, list(current_models), xopt))

        ind_worst_model = np.argsort(xopt)[0]
        worst_model = current_models.pop(ind_worst_model)
        print "Remove Model {}".format(worst_model)

        iteration += 1

        break

    score_hist.sort()
    best_models = score_hist[-1][1]
    best_xopts = score_hist[-1][2]
    
    retxopt = np.zeros(nmodels)
    for ixopt, imodel in enumerate(best_models) :
        retxopt[imodel] = best_xopts[ixopt]

    print retxopt

    return retxopt

def train_level1_linreg(X_train, y) :

    model = linear_model.LinearRegression()
    model.fit(X_train, y)
    pred = model.predict(X_train)
    return model.coef_

def main() :

    level1fn = 'logreg_level1data_rev{}.dat'.format(utility.logregrev)
    X_train = np.load(utility.ddir + '/' + level1fn)
    y = np.array(utility.load_truth(), dtype=np.float64)

    weightfn = 'logreg_level1weights_rev{}.dat'.format(utility.logregrev)
    weights = train_level1(X_train, y)
    weights.dump(utility.ddir + '/' + weightfn)

    lgwfn = 'logreg_level1weights_linreg_rev{}.dat'.format(utility.logregrev)
    lgweights = train_level1_linreg(X_train, y)
    lgweights.dump(utility.ddir + '/' + lgwfn)
    
if __name__ == "__main__":
    main()
