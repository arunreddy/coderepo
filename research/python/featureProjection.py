'''
 Implementation of Feature Projection algorithm from the paper Wang, Chang, and Sridhar Mahadevan. “Heterogeneous Domain Adaptation Using Manifold Alignment.” IJCAI Proceedings-International Joint Conference on Artificial Intelligence. Vol. 22. N.p., 2011. 1541. Print.
''' 

import os
import sys
import numpy as np
import nltk
import os
import os.path
import scipy as sp

import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import pairwise_distances
import itertools
import pyprind
import psutil
import joblib
from collections import namedtuple

'''
Arguments:
  src: Source Domain path
  tgt: Target Domain path
  n: size of the docs from each domain to consider for the task.
  k: svd component size
'''
fileName,src,tgt,n =sys.argv


'''
    Functions
'''

def getCorpus(lines):
    ids = list()
    labels = list()
    corpus = list()
    for line in lines:
        strs=line.rstrip('\n').split("\t",2)
        ids.append(strs[0])
        labels.append(strs[1])
        corpus.append(strs[2])
    return ids,labels,corpus

def loadMultiDomainData(domain):
    posFile = "{0}/data/domains/{1}/positive.review.text".format(os.environ['HOME'],domain)
    negFile = "{0}/data/domains/{1}/negative.review.text".format(os.environ['HOME'],domain)
    
    posLines = tuple(open(posFile,'r'))
    negLines = tuple(open(negFile,'r'))

    
    p_ids,p_labels,p_corpus = getCorpus(posLines)
    n_ids,n_labels,n_corpus = getCorpus(negLines)

    return p_ids+n_ids,p_labels+n_labels,p_corpus+n_corpus
    
def classConditionalProbs(X,Y,classLabels,smoothing='None'):
    noOfFeatures=X.shape[1]
    Xprime = np.ndarray((len(classLabels),noOfFeatures),dtype=float)
    
    if smoothing == 'additive':
        for c in classLabels:
            indices = [i for i, x in enumerate(Y) if x == c]
            Xprime[c,:]=(X[indices,:].sum(axis=0)+1)/(len(indices)+noOfFeatures)
    else:
        for c in classLabels:
            indices = [i for i, x in enumerate(Y) if x == c]
            Xprime[c,:]=X[indices,:].sum(axis=0)/len(indices)

    return Xprime;

def symKlDivergence(P,Q,classLabels):
    kldC=0.0;
    for c in classLabels:
        kldST=sp.stats.entropy(P[c,:],Q[c,:])
        kldTS=sp.stats.entropy(Q[c,:],P[c,:])
        kldC+=(kldST+kldTS);
    
    return kldC;

def klDivergence(XS,XT):
    A=(XS.sum(axis=0).transpose()+1)/(XS.shape[0]+XS.shape[1])
    B=(XT.sum(axis=0).transpose()+1)/(XT.shape[0]+XT.shape[1])
    KAB=sp.stats.entropy(A,B)
    KBA=sp.stats.entropy(B,A)
    return KAB+KBA;

# TODO: Speed up the computation
def generate_WS_WD_matrix(A,B):
    nA = len(A);
    nB = len(B);
    simMat = np.ndarray((nA+nB,nA+nB),dtype=float)
    diffMat = np.ndarray((nA+nB,nA+nB),dtype=float)
    C = A+B;
    nC=nA+nB;
    
    ticks = (nC*(nC+1)/2);
    #bar = pyprind.ProgBar(ticks,monitor=True)
    for c in itertools.combinations(range(nC),2):
    #        bar.update()
            i=c[0];
            j=c[1];
            if(C[i]==C[j]):
                simMat[i,j]=1.0;
            else:
                diffMat[i,j]=1.0;
    
    simMat = sp.sparse.csc_matrix(simMat+simMat.transpose()+np.eye(nC))
    diffMat = sp.sparse.csc_matrix(diffMat+diffMat.transpose())
    #print(bar)    
    return simMat,diffMat;


def generate_topology_matrix(A,B):
    #nA = A.shape[0];
    #nB = B.shape[0];
    #tMat = np.ndarray((nA+nB,nA+nB),dtype=float)
    C = sp.sparse.vstack((A, B), format='csr');
    CD = pairwise_distances(C,n_jobs=-1);
    #print(CD.shape)
    CD=np.exp(np.square(CD)*-1)+np.eye(CD.shape[0])
    return CD
    
    #nC=nA+nB;
    
    #ticks = (nC*(nC+1)/2);
    #bar = pyprind.ProgBar(ticks,monitor=True)
    #for i in itertools.combinations(range(nA+nB),2):
        #bar.update()
    #    xa = C[i[0],:]
    #    xb = C[i[1],:]
    #    dist=pairwise_distances(xa,xb,n_jobs=-1)[0][0]
        #tMat[i[0],i[1]]=math.exp(-1*dist*dist)    
    
    
    #tMat = sp.sparse.csc_matrix(tMat+tMat.transpose())
    #print(bar)
    #return tMat;

def laplacian_matrix(A):
    D=sp.sparse.csc_matrix(A.sum(axis=0)).multiply(sp.sparse.eye(A.shape[0]));
    return D-A;

def reweighting_scheme_01():
    pass





'''
MAIN
'''

# Load the data
s_ids,s_labels,s_corpus=loadMultiDomainData(src)
t_ids,t_labels,t_corpus=loadMultiDomainData(tgt)

subSetSize = int(n);
data_s = s_corpus[0:0+subSetSize]+s_corpus[1000:1000+subSetSize]
data_t = t_corpus[0:0+subSetSize]+t_corpus[1000:1000+subSetSize]

vectorizer = CountVectorizer(min_df=1,stop_words='english',binary=True)
X=vectorizer.fit_transform(data_s+data_t);
Y = [1]*subSetSize+[0]*subSetSize+[1]*subSetSize+[0]*subSetSize

XS=sp.sparse.csc_matrix(X[0:2*subSetSize,:],dtype=float)
XT=sp.sparse.csc_matrix(X[2*subSetSize:4*subSetSize,:],dtype=float)
YS=Y[0:2*subSetSize]
YT=Y[2*subSetSize:4*subSetSize]

# Concatenate source and target.
XF=sp.sparse.vstack((XS,XT), format='csc')

# Generate similar and dissimilar matrices
WS,WD=generate_WS_WD_matrix(YS,YT);
LS = laplacian_matrix(WS)
LD = laplacian_matrix(WD)


scoreList=[];

# On original
clf1 = svm.SVC(kernel='linear', C=1)
clf1.fit(XS,YS);
pred = clf1.predict(XT);
Score=namedtuple('Score',['t','s'])
scoreList.append(Score('original',accuracy_score(YT,pred)))
scoreList.append(Score('original_kl',klDivergence(XS,XT)))

for p in range(50,550,50):
    # Apply SVD and find the reduced subspace.
    U,S,VT=sp.sparse.linalg.svds(XF,k=p)

    # Instances in the latent space.
    U=sp.sparse.csc_matrix(U,dtype=float)
    XS_svd=U[0:2*subSetSize,:]
    XT_svd=U[2*subSetSize:4*subSetSize,:]

    T=generate_topology_matrix(XS_svd,XT_svd)
    L = laplacian_matrix(T)

    # Create Z matrix
    XS_zero=sp.sparse.csc_matrix(XS_svd.transpose().shape,dtype=float)
    XT_zero=sp.sparse.csc_matrix(XT_svd.transpose().shape,dtype=float)
    s1=sp.sparse.vstack((XS_svd.transpose(),XT_zero), format='csc')
    s2=sp.sparse.vstack((XT_zero,XT_svd.transpose()), format='csc')
    Z =sp.sparse.hstack((s1,s2), format='csc')


    # Objective function 
    A = Z*(LS+L)*Z.transpose();
    B = Z*LD*Z.transpose();

    # Calculating the eigen vectors.
    eigVal,eigVec=sp.sparse.linalg.eigsh(A,M=B,k=p,which='SM')
    #print("Shape of scalar eigen values",eigVal.shape)

    # Projection function
    FS=eigVec[:p]
    FT=eigVec[p:]

    # Projection
    XS_proj=XS_svd*FS;
    XT_proj=XT_svd*FT;
    #print("Shape of XA proj",XS_proj.shape);
    #print("Shape of XA proj",XT_proj.shape);

    # SVD Linear kernel

    # After SVD
    clf2 = svm.SVC(kernel='linear', C=1)
    clf2.fit(XS_svd,YS);
    pred = clf2.predict(XT_svd);
    scoreList.append(Score('svd_%d'%p,accuracy_score(YT,pred)))
    scoreList.append(Score('svd_%d_kl'%p,klDivergence(XS_svd,XT_svd)))


    # On projected plane
    clf3 = svm.SVC(kernel='linear', C=1)
    clf3.fit(XS_proj,YS);
    pred = clf3.predict(XT_proj);
    scoreList.append(Score('proj_%d'%p,accuracy_score(YT,pred)))
    scoreList.append(Score('proj_%d_kl'%p,klDivergence(XS_proj,XT_proj)))


# Save the file to disk.
fileName="%s_%s"%(src,tgt)
joblib.dump(scoreList,fileName)

