import os
import pickle
import utility
import numpy as np
from scipy import sparse
from sklearn import metrics, cross_validation, linear_model

def pass_over_data(features, Xts, y, model, cvgen) :

    print "Executing a full pass..."
    scores = []
    for f in features :
        feats = [feat for feat in features if feat != f]
        Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
        cvscores = cross_validation.cross_val_score(
            model, Xt, y, cv=cvgen, n_jobs=4,
            scoring='roc_auc')
        score = cvscores.mean()
        scores.append((score, f))
        print "(Full Pass) Feature {} Mean AUC: {}".format(f, score)
    
    return sorted(scores)

def get_scores(npass, seed, features, Xts, y, model, cvgen) :

    print "Getting scores..."
    scores = None
    scoresfn = 'feateng_seed{}_pass{}.pickle'.format(seed, npass)
    scoresfn = utility.ddir + '/' + scoresfn
    if os.path.exists(scoresfn) :
        scores = pickle.load(open(scoresfn))
    else :
        scores = pass_over_data(features, Xts, y, model, cvgen)
        pickle.dump(scores, open(scoresfn, 'w'))
    return scores

def backward_feature_selection(seed, X_train, y, Xts) :
    
    ntrain = X_train.shape[0]
    nfeat = X_train.shape[1]

    print "Setting up the model..."
    newpredict = lambda(self, X) : self.predict_proba(X)[:,1]
    model = linear_model.LogisticRegression()
    model.predict = newpredict
    model.C = 1.485994

    print "Setting up the cross validation..."
    N = 10
    cvgen = cross_validation.ShuffleSplit(ntrain, N, 0.2, random_state=seed)

    print "Getting full pass results..."
    current_features = set(range(nfeat))
    scores = get_scores(0, seed, current_features, Xts, y, model, cvgen)

    print "Starting greedy selection loop..."
    iteration = 0
    score_hist = []
    while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
        Xt = sparse.hstack([Xts[j] for j in current_features]).tocsr()
        cvscores = cross_validation.cross_val_score(
            model, Xt, y, cv=cvgen, n_jobs=4,
            scoring='roc_auc')
        score = cvscores.mean()
        print "Iteration {} Mean AUC: {}".format(iteration, score)

        worst_feature = scores.pop(-1)[1]
        current_features.remove(worst_feature)
        print "Remove Feature {}".format(worst_feature)

        score_hist.append((score, worst_feature))
        iteration += 1

        npassover = 10
        if iteration % npassover == 0 :
            npass = int(iteration / float(npassover))
            scores = get_scores(npass, seed, current_features,
                                Xts, y, model, cvgen)
    
    current_features.add(score_hist[-1][1])
    features = np.array(list(current_features))
    print "Selected features {}: {}".format(seed, features)

    return features

def main() :

    print "Loading data..."
    X_train = utility.load_encoded('train')
    y = utility.load_truth()

    print "Loading indexing..."
    xtsfn = '{}/logreg_xts.pickle'.format(utility.ddir)
    if not os.path.exists(xtsfn) :
        Xts = [utility.OneHotEncoder(X_train[:,[i]])[0] for i in range(nfeat)]
        pickle.dump(Xts, open(xtsfn, "w"))
    else :
        Xts = pickle.load(open(xtsfn))

    for iseed in range(5) :
        seedfn = '{}/feateng_backward_seed{}.dat'.format(utility.ddir, iseed)
        if os.path.exists(seedfn) : continue

        features = backward_feature_selection(iseed, X_train, y, Xts)
        features.dump(seedfn)
    
if __name__ == "__main__":
    main()
