import os
import pickle
import utility
import numpy as np
from scipy.io import savemat
from sklearn.metrics import auc_score
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

def main() :

    y = utility.load_truth()
    X_train = utility.load_encoded('train')

    good_features = [0, 8, 9, 10, 19, 34, 36, 37, 38, 41, 42, 43, 47, 53,
                     55, 60, 61, 63, 64, 67, 69, 71, 75, 81, 82, 85]

    X_train, keymap = utility.OneHotEncoder(X_train[:,good_features])

    ntrain = X_train.shape[0]

    nb_cvscores = []
    lgr_cvscores = []
    combined_cvscores = []
    cvgen = KFold(ntrain, 10, random_state=utility.SEED)
    for train_inds, test_inds in cvgen :

        X_cvtrain = X_train[train_inds]
        X_cvtest = X_train[test_inds]
        y_cvtrain = y[train_inds]
        y_cvtest = y[test_inds]

        # Fit the Bayesian Classifier
        mb = MultinomialNB()
        mb.fit(X_cvtrain, y_cvtrain)
        mbpred_cvtrain = mb.predict_proba(X_cvtrain)[:,1]

        lgr = LogisticRegression()
        lgr.fit(np.reshape(mbpred_cvtrain, (len(train_inds), 1)), y_cvtrain)

        # Predict the training data
        mbpred_cvtest = mb.predict_proba(X_cvtest)[:,1]
        mbpred_cvtest = np.reshape(mbpred_cvtest, (len(test_inds), 1))
        nb_pred_cvtest = lgr.predict_proba(mbpred_cvtest)[:,1]

        # Logistic Regression Only
        lgrmodel = LogisticRegression()
        lgrmodel.fit(X_cvtrain, y_cvtrain)
        lgr_pred_cvtest = lgrmodel.predict_proba(X_cvtest)[:,1]

        # Combined
        combined_pred_cvtest = np.mean(
            np.vstack((nb_pred_cvtest, lgr_pred_cvtest)), axis=0)

        # Recored Scores
        print
        nb_cvscore = auc_score(y_cvtest, nb_pred_cvtest)
        nb_cvscores.append(nb_cvscore)
        print nb_cvscore

        lgr_cvscore = auc_score(y_cvtest, lgr_pred_cvtest)
        lgr_cvscores.append(lgr_cvscore)
        print lgr_cvscore
        
        combined_cvscore = auc_score(y_cvtest, combined_pred_cvtest)
        combined_cvscores.append(combined_cvscore)
        print combined_cvscore

    print np.mean(nb_cvscores)
    print np.mean(lgr_cvscores)
    print np.mean(combined_cvscores)

if __name__ == '__main__' :
    main()
