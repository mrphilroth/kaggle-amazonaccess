import itertools
import collections
import numpy as np
import pylab as plt
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite, mmread
from sklearn.metrics import roc_curve, auc
from os.path import dirname, abspath, exists, realpath

"""
Current: 0.90868 (132)
0.90716 for top 10% (168)

Logistic Regression Feature Selection:
Greedy feature selection on 92 pre-encoded features
 Add element of random-ness and ensemble
 Try backward greedy feature extraction with random elements
 Expand to 4-tuples with rare event column?

Other models:
percentage granted of various categories, features from results
  Cross validation is necessary
vowpal wabbit
pybrain
libfm

Done:
Gradient boosting machines /requires dense arrays

stacking of these methods
"""

bdir = dirname(realpath(__file__))
ddir = abspath(bdir + "/../data/")
subdir = abspath(bdir + "/../submissions/")
if not exists(ddir) : os.mkdir(ddir)
if not exists(subdir) : os.mkdir(subdir)

SEED = 25
gbmrev = 1
logregrev = 4

def plot_roc_curve(fn, fpr, tpr, auc_score) :
    """ 
    Creates a plot of the receiver operator characteristic.
    """
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(fn)

def eval_auc(truth, pred, fn=None) :
    """ 
    A quick interface to sklearn's method for evaluating the area
    under a receiver operator curve.
    """
    fpr, tpr, thresholds = roc_curve(truth, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    if fn : plot_roc_curve(fn, fpr, tpr, auc_score)
    return auc_score

def load_dataframe(s='train') :
    if not s in ['train', 'test'] :
        print "Don't recognize data set {}".format(s)
        return None
    df = pd.io.parsers.read_csv('{}/{}.csv'.format(ddir, s))
    return df

def OneHotEncoder(data, keymap=None) :
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None :
          keymap = []
          for col in data.T :
               uniques = set(list(col))
               if 0 in uniques : uniques.remove(0)
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T) :
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col) :
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def remove_rare(data, threshold=40) :
    """
    Remove rare features from the data set by setting them to zero.
    """
    ndata = data.shape[0]
    nfeat = data.shape[1]
    data = np.ravel(data)

    bc = collections.defaultdict(int)
    for val in data : bc[val] += 1

    bckeys = bc.keys()
    for k in bckeys :
        if bc[k] < threshold :
            del bc[k]

    validkeys = np.array(bc.keys())
    mask = np.logical_not(np.in1d(data.flatten(), validkeys))
    data[mask] = 0
    data = np.reshape(data, (ndata, nfeat))

    cols_with_uniques = []
    for i, col in enumerate(data.T) :
        if np.sum(col != 0) > 0 :
            cols_with_uniques.append(i)

    return data[:,cols_with_uniques]

def count_unique(keys) :
    """
    Return number of occurrences of each element
    """
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return zip(np.bincount(bins), uniq_keys)

def group_data(data, degree=3, hash=hash, remove_unique=False):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m, n = data.shape
    for indicies in itertools.combinations(range(n), degree) :
        col = [hash(tuple(v)) for v in data[:,indicies]]
        if remove_unique :
            countdict = count_unique(col)
            badhashes = set([hashv for count, hashv in countdict if count == 1])
            for i in range(len(col)) :
                if col[i] in badhashes :
                    badhashes.remove(col[i])
                    col[i] = 0
        new_data.append(col)
    return np.array(new_data).T

def encode() :
    """
    Generate extra features from pairs, triplets, and common
    quadruplets of the existing features and then save those features
    in a sparse matrix to disk.
    """
    dftrain = load_dataframe('train')
    dftest = load_dataframe('test')
    lentrain = len(dftrain)
    all_data = np.vstack((dftrain.ix[:,1:-1], dftest.ix[:,1:-1]))
    np.array(dftrain.ACTION).dump('{}/train_truth.dat'.format(ddir))
    
    dp = group_data(all_data, degree=2, remove_unique=True)
    dt = group_data(all_data, degree=3, remove_unique=True)
    dq = group_data(all_data, degree=4, remove_unique=True)
    dq = remove_rare(dq, 15)

    X = all_data[:lentrain]
    X_2 = dp[:lentrain]
    X_3 = dt[:lentrain]
    X_4 = dq[:lentrain]
    X_train_all = np.hstack((X, X_2, X_3, X_4))
    mmwrite('{}/train_encoded'.format(ddir), X_train_all)

    X_test = all_data[lentrain:]
    X_test_2 = dp[lentrain:]
    X_test_3 = dt[lentrain:]
    X_test_4 = dq[lentrain:]
    X_test_all = np.hstack((X_test, X_test_2, X_test_3, X_test_4))
    mmwrite('{}/test_encoded'.format(ddir), X_test_all)

def load_truth() :
    fn = '{}/train_truth.dat'.format(ddir)
    if not exists(fn) : encode()
    return np.load(fn)

def load_encoded(s='train') :
    if not s in ['train', 'test'] :
        print "Don't recognize data set {}".format(s)
        return None

    fn = '{}/{}_encoded.mtx'.format(ddir, s)
    if not exists(fn) : encode()
    return mmread(fn)

def create_test_submission(filename, prediction):
    filename = subdir + "/" + filename
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('{},{}'.format(i+1, p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()

