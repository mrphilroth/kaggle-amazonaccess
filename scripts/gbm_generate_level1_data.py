import os
import pickle
import utility
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics, cross_validation, linear_model

def model_generate_level1(bestc, features, X_train, y) :

    ntrain = X_train.shape[0]
    newdata = np.zeros(ntrain)

    X_train, keymap = utility.OneHotEncoder(X_train[:,features])

    mbmodel = MultinomialNB()
    mb.fit(X_cvtrain, y_cvtrain)
    mbpred_cvtrain = mb.predict_proba(X_cvtrain)[:,1]

    lgrmodel = LogisticRegression()
    lgr.fit(np.reshape(mbpred_cvtrain, (len(train_inds), 1)), y_cvtrain)


    model = GradientBoostingRegressor(loss='ls', learning_rate=0.1,
                                      n_estimators=1000, max_depth=4,
                                      verbose=1)

    cvscores = []
    cvgen = cross_validation.KFold(ntrain, 10, random_state=utility.SEED)
    for train_inds, test_inds in cvgen :
        X_cvtrain = X_train[train_inds]
        X_cvtest = X_train[test_inds]
        y_cvtrain = y[train_inds]
        y_cvtest = y[test_inds]

        model.fit(X_cvtrain.toarray(), y_cvtrain)
        pred_cvtest = model.predict(X_cvtest)
        cvscore = metrics.auc_score(y_cvtest, pred_cvtest)
        cvscores.append(cvscore)

        newdata[test_inds] = pred_cvtest

    print "Average CV Score: {}".format(np.mean(cvscores))
    return newdata

def generate_level1(mlist, X_train, y) :

    level1_data = None
    for modelstr in mlist :
        print modelstr

        featfn = '{}/{}.dat'.format(utility.ddir, modelstr)
        paramfn = featfn.replace('.dat', '_bestc.txt')

        features = np.load(featfn)
        bestc = float(open(paramfn, 'r').read().strip())

        newdata = model_generate_level1(bestc, features, X_train, y)

        if level1_data == None :
            level1_data = newdata
        else :
            level1_data = np.vstack((level1_data, newdata))

    return np.transpose(level1_data)

def main() :

    print "Loading data..."
    X_train = utility.load_encoded('train')
    y = utility.load_truth()

    mlist = []
    for featfn in os.listdir(utility.ddir) :
        if not (featfn.startswith('feateng') and
                featfn.endswith('dat')) : continue
        modelstr = os.path.splitext(featfn)[0]
        featfn = utility.ddir + '/' + featfn
        paramfn = featfn.replace('.dat', '_bestc.txt')
        if not os.path.exists(paramfn) : continue

        mlist.append(modelstr)

    mlist = [mlist[0]]
    
    level1_data = generate_level1(mlist, X_train, y)
    level1fn = 'gbm_level1data_rev{}.dat'.format(utility.gbmrev)
    level1_data.dump(utility.ddir + '/' + level1fn)
    
if __name__ == '__main__' :
    main()
