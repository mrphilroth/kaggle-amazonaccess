import os
import pickle
import utility
import numpy as np
from scipy import sparse
from sklearn import metrics, cross_validation, linear_model

nrand = 2

def forward_feature_selection(seed, X_train, X_test, y, Xts) :
    
    ntrain = X_train.shape[0]
    nfeat = X_train.shape[1]

    newpredict = lambda(self, X) : self.predict_proba(X)[:,1]
    model = linear_model.LogisticRegression()
    model.predict = newpredict
    model.C = 1.485994
    N = 10
    score_hist = []
    good_features = set([])
    np.random.seed(seed)
    topick = np.floor(np.random.uniform(0, nrand, nfeat)) + 1
    cvgen = cross_validation.ShuffleSplit(ntrain, N, 0.2, random_state=seed)
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
         scores = []
         for f in range(len(Xts)):
             if f not in good_features:
                 feats = list(good_features) + [f]
                 Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                 cvscores = cross_validation.cross_val_score(
                     model, Xt, y, cv=cvgen, n_jobs=4,
                     scoring='roc_auc')
                 score = cvscores.mean()
                 scores.append((score, f))
                 print "Feature: %i Mean AUC: %f" % (f, score)
         
         ipick = int(topick[len(good_features)])
         good_features.add(sorted(scores)[-ipick][1])
         score_hist.append(sorted(scores)[-ipick])
         print "Current features: %s" % sorted(list(good_features))
    
    # Remove last added feature from good_features
    good_features.remove(score_hist[-1][1])
    good_features = sorted(list(good_features))

    print "Selected features {}: {}".format(seed, good_features)
    seedfn = '{}/logreg_nrand{}_seed{}.dat'.format(utility.ddir, nrand, seed)
    np.array(good_features).dump(seedfn)

def main() :

    X_train = utility.load_encoded('train')
    X_test = utility.load_encoded('test')
    y = utility.load_truth()

    xtsfn = '{}/logreg_xts.pickle'.format(utility.ddir)
    if not os.path.exists(xtsfn) :
        Xts = [utility.OneHotEncoder(X_train[:,[i]])[0] for i in range(nfeat)]
        pickle.dump(Xts, open(xtsfn, "w"))
    else :
        Xts = pickle.load(open(xtsfn))

    for iseed in range(6) :
        forward_feature_selection(iseed, X_train, X_test, y, Xts)
    
if __name__ == "__main__":
    main()
