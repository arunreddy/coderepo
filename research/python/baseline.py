from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import scipy.sparse as sp
import sys

src=sys.argv[1]
tgt=sys.argv[2]

# Functions
def getCorpus(lines):
    corpus = list()
    for line in lines:
        content=line.rstrip('\n').split("\t")[2]
        corpus.append(line)
    return corpus

def loadData(domain):
    posFile = "/home/arun/data/pod/domains/{0}/positive.review.text".format(domain)
    negFile = "/home/arun/data/pod/domains/{0}/negative.review.text".format(domain)

    posLines = tuple(open(posFile,'r'))
    negLines = tuple(open(negFile,'r'))
    
    posCorpus = getCorpus(posLines)
    negCorpus = getCorpus(negLines)

    return posCorpus+negCorpus


srcCorpus = loadData(src)
tgtCorpus = loadData(tgt)



# Extract features from text
vectorizer = CountVectorizer(min_df=1,stop_words='english',binary=True)
#binaryVectorizer = CountVectorizer(min_df=1,stop_words='english',binary=True)
#tfIdfVectorizer = TfidfVectorizer(min_df=1,stop_words='english')

Xarr=vectorizer.fit_transform(srcCorpus+tgtCorpus);
Y = [1]*1000+[0]*1000+[1]*1000+[0]*1000

# Data
X_src = Xarr[0:2000,:];
X_tgt = Xarr[2000:4000,:];
Y_src = Y[0:2000];
Y_tgt = Y[2000:4000];



# Source Only
score_a=0;
clf1 = svm.SVC(kernel='linear', C=1)
clf1.fit(X_src,Y_src);
pred = clf1.predict(X_tgt);
score_a=accuracy_score(Y_tgt,pred)
  
# Source+Target combined.
score_b=0;
clf2 = svm.SVC(kernel='linear', C=1)
clf2.fit(Xarr,Y);
pred = clf2.predict(X_tgt);
score_b=accuracy_score(Y_tgt,pred)


def stratified(n_folds):
    scores_arr=[]
    skf = StratifiedKFold(Y_tgt, n_folds)
    cnt=0;
    for train_index, test_index in skf:
        clf3 = svm.SVC(kernel='linear', C=1)
        X_tgt_train = X_tgt[test_index]
        X_tgt_test = X_tgt[train_index]
        Y_tgt_array = np.asarray(Y_tgt);
        Y_tgt_train = Y_tgt_array[test_index]
        Y_tgt_test = tuple(Y_tgt_array[train_index])

        A=sp.vstack((X_src, X_tgt_train), format='csr');
        B=Y_src+Y_tgt_train.tolist();
        clf3.fit(A,B);
        pred = clf3.predict(X_tgt_test);
        acc =accuracy_score(Y_tgt_test,pred)
        scores_arr.append(acc)
        cnt=cnt+1
        if cnt > 10:
            break;

    scores_str="{0}±{1}\t".format(np.round(np.mean(scores_arr),3),np.round(np.var(scores_arr),2))
    return scores_str



score_c=stratified(500)
score_d=stratified(100)
score_e=stratified(50)


print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}".format(src,tgt,score_a,score_b,score_c,score_d,score_e))
    
'''
# Using Cross Validation
scores = cross_validation.cross_val_score(clf, X1, Y, cv=10)
print(scores)

scores = cross_validation.cross_val_score(clf, X2, Y, cv=10)
print(scores)

scores = cross_validation.cross_val_score(clf, X3, Y, cv=10)
print(scores)
'''
