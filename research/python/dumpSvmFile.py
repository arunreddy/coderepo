from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import scipy.sparse as sp
import sys

from sklearn.datasets import dump_svmlight_file

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

fileStr="{0}_{1}".format(src,tgt)
dump_svmlight_file(X_src,Y_src,fileStr+".src")
dump_svmlight_file(X_src,Y_src,fileStr+".tgt")
