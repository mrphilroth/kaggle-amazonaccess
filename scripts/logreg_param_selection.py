import os
import pickle
import utility
import numpy as np
from scipy import sparse
from sklearn import metrics, cross_validation, linear_model

def get_best_hyperparam(features, Xts, y) :
    
    print "Initializing"
    score_hist = []
    Xt = sparse.hstack([Xts[j] for j in features]).tocsr()
    ntrain = Xt.shape[0]

    print "Setting up the logistic regression model"
    newpredict = lambda(self, X) : self.predict_proba(X)[:,1]
    model = linear_model.LogisticRegression()
    model.predict = newpredict

    print "Finding the best regularization parameter"
    Cvals = np.logspace(-4, 4, 32, base=2)
    cvgen = cross_validation.ShuffleSplit(
        ntrain, 10, 0.2, random_state=utility.SEED)
    for C in Cvals:
        model.C = C
        cvscores = cross_validation.cross_val_score(
            model, Xt, y, cv=cvgen, n_jobs=4,
            scoring='roc_auc')
        score = cvscores.mean()
        score_hist.append((score, C))
        # print "C: %f Mean AUC: %f" % (C, score)

    bestc = sorted(score_hist)[-1][1]
    print "Best C value: %f with score %f" % (bestc, sorted(score_hist)[-1][0])
    return bestc

def main() :

    print "Loading data..."
    y = utility.load_truth()

    print "Loading indexing..."
    Xts = None
    xtsfn = '{}/logreg_xts.pickle'.format(utility.ddir)
    if not os.path.exists(xtsfn) :
        X_train = utility.load_encoded('train')
        nfeat = X_train.shape[1]
        Xts = [utility.OneHotEncoder(X_train[:,[i]])[0] for i in range(nfeat)]
        pickle.dump(Xts, open(xtsfn, "w"))
    else :
        Xts = pickle.load(open(xtsfn))

    mlist = []
    for featfn in os.listdir(utility.ddir) :
        if not (featfn.startswith('feateng') and
                featfn.endswith('dat')) : continue
        modelstr = os.path.splitext(featfn)[0]
        featfn = utility.ddir + '/' + featfn
        paramfn = featfn.replace('.dat', '_bestc.txt')
        if os.path.exists(paramfn) : continue

        print modelstr

        features = np.load(featfn)
        bestC = get_best_hyperparam(features, Xts, y)
        ofile = open(paramfn, 'w')
        ofile.write('{}\n'.format(bestC))
        ofile.close()
    
if __name__ == "__main__":
    main()
