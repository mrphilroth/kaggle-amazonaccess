import utility
import numpy as np
import pandas as pd
from scipy import sparse
from numpy import array, hstack
from itertools import combinations
from sklearn import metrics, cross_validation, linear_model

SEED = 25

def main() :

    print "Reading dataset..."
    X_train_all = utility.load_encoded('train')
    X_test_all = utility.load_encoded('test')
    y = utility.load_truth()

    num_train = X_train_all.shape[0]
    num_feat = X_train_all.shape[1]

    print "Loading indexing..."
    Xts = [utility.OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_feat)]

    print "Setting up the model..."
    newpredict = lambda(self, X) : self.predict_proba(X)[:,1]
    model = linear_model.LogisticRegression()
    model.predict = newpredict

    # good_features = [0, 8, 9, 10, 19, 34, 36, 37, 38, 41, 42,
    #                  43, 47, 53, 55, 60, 61, 63, 64, 67, 69,
    #                  71, 75, 81, 82, 85, 97, 103, 105, 111, 
    #                  112, 114, 125, 127]
    # model.C = 1.30775906845

    good_features = [0,  8,  9, 10, 19, 34, 36, 37, 38, 41, 42, 43,
                     47, 53, 55, 60, 61, 63, 64, 67, 69, 71, 75, 81,
                     82, 85, 97, 103, 105, 108, 115, 122, 141]
    model.C = 1.30775906845

    # good_features = [0, 8, 9, 10, 19, 34, 36, 37, 38, 41, 42, 43, 47, 53,
    #                  55, 60, 61, 63, 64, 67, 69, 71, 75, 81, 82, 85]
    # model.C = 1.485994

    # good_features = [ 0,  7,  8, 29, 42, 63, 64, 67, 69, 85]
    # model.C = 1.09355990876    
    print "Selected features %s" % good_features
    
    print "Getting a CV score..."
    N = 10
    cvgen = cross_validation.ShuffleSplit(num_train, N, 0.2, random_state=25)
    Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
    cvscores = cross_validation.cross_val_score(
        model, Xt, y, cv=cvgen, n_jobs=4,
        scoring='roc_auc')
    score = cvscores.mean()
    print "Mean CV score: {}".format(score)

    print "Performing One Hot Encoding on entire dataset..."
    Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
    Xt, keymap = utility.OneHotEncoder(Xt)
    X_train = Xt[:num_train]
    X_test = Xt[num_train:]
    
    print "Training full model..."
    model.fit(X_train, y)
    
    print "Making prediction and saving results..."
    preds = model.predict_proba(X_test)[:,1]
    submitfn = 'logistic_regression_pred_headstart1.csv'
    utility.create_test_submission(submitfn, preds)
    
if __name__ == "__main__":
    main()
    
