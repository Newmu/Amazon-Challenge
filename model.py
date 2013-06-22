import numpy as np
import csv
from scipy.stats import gaussian_kde as kde
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2,RFE
from itertools import combinations,permutations
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import cPickle
from matplotlib import pyplot as plt
from os import system,getpid
from copy import deepcopy
from time import time

SEED = 42
system("taskset -p 0xff %d" % getpid())

def KDE(a,evalRange=None,bins=1000):
	assert len(a) > 1, "Can't KDE a single value"
	if evalRange is 'padded':
		padding = np.max(a)-np.min(a)
		x = np.linspace(np.min(a)-padding,np.max(a)+padding,bins)
	elif evalRange is None:
		x = np.linspace(np.min(a),np.max(a),bins)
	else:
		x = np.linspace(evalRange[0],evalRange[1],bins)
	pdf = kde(a)
	y = pdf(x)
	return x,y

def load(path):
	reader = csv.reader(open(path),delimiter=',', skipinitialspace=True)
	X = []
	Y = []
	for i,line in enumerate(reader):
		if i == 0:
			fields = line
		else:
			x = [int(line[n]) for n in range(1,len(line)-1)]
			X.append(x)
			Y.append(int(line[0]))
	return np.array(X),np.array(Y)

def loadTest(path):
	reader = csv.reader(open(path),delimiter=',', skipinitialspace=True)
	X = []
	for i,line in enumerate(reader):
		if i == 0:
			fields = line
		else:
			x = [int(line[n]) for n in range(1,len(line)-1)]
			X.append(x)
	return np.array(X)

def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def group_data(data, degree=2, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
    	if 5 in set(indicies) and 7 in set(indicies):
    		print 'passing',indicies
    	# elif 2 in set(indicies) and 3 in set(indicies):
    	# 	print 'passing',indicies
    	else:
    		# print indicies
    		new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    	# new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return np.array(new_data).T

def cv(clf,X,Y,n=10,proba=True):
	aucs = []
	for i in range(n):
		X_train, X_cv, y_train, y_cv = train_test_split(X, Y, test_size=.20, random_state=i*SEED)
		# t = time()
		clf.fit(X_train, y_train)
		# print time()-t
		if proba: 
			preds = clf.predict_proba(X_cv)[:, 1]
		else:
			preds = clf.predict(X_cv)
		fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
		auc = metrics.auc(fpr, tpr)
		# print "AUC (fold %d/%d): %f" % (i + 1, n, auc)
		aucs.append(auc)
	return np.mean(auc)

def cv2(clf,clf2,X,Y,n=10,proba=True,proba2=True):
	aucs = []
	for i in range(n):
		X_train, X_cv, y_train, y_cv = train_test_split(X, Y, test_size=.20, random_state=i*SEED)
		t = time()
		clf.fit(X_train, y_train)
		print time()-t
		t = time()
		clf2.fit(X_train,y_train)
		print time()-t
		if proba: 
			preds = clf.predict_proba(X_cv)[:, 1]
		else:
			preds = clf.predict(X_cv)
		if proba2:
			preds2 = clf2.predict_proba(X_cv)[:, 1]
		else:
			preds2 = clf2.predict(X_cv)
		preds = (preds*2.+preds2)/3.
		fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
		auc = metrics.auc(fpr, tpr)
		print "AUC (fold %d/%d): %f" % (i + 1, n, auc)
		aucs.append(auc)
	return np.mean(auc)

X,Y = load('data/train.csv')
nTrain = len(Y)
testX = loadTest('data/test.csv')
order1Combs = np.vstack((X,testX))
order2Combs = group_data(order1Combs,degree=2)
order3Combs = group_data(order1Combs,degree=3)
order4Combs = group_data(order1Combs,degree=4)
order5Combs = group_data(order1Combs,degree=5)
# order6Combs = group_data(order1Combs,degree=6)
# allCombs = np.hstack((order1Combs,order2Combs,order3Combs,order4Combs,order5Combs,order6Combs))
allCombs = np.hstack((order1Combs,order2Combs,order3Combs,order4Combs,order5Combs))
# allCombs = np.hstack((order1Combs,order2Combs,order3Combs,order4Combs))
# allCombs = np.hstack((order1Combs,order2Combs,order3Combs))
# allCombs = np.hstack((order1Combs,order2Combs))
# allCombs = order1Combs
allCombsDictList = [dict(zip(range(len(row)),[str(field) for field in row])) for row in allCombs]
vect = DictVectorizer()
allX = vect.fit_transform(allCombsDictList)
# t = time()
# pca = TruncatedSVD(n_components=200)
# allX = pca.fit_transform(allX)
# print time()-t
trainX = allX[:nTrain]
testX = allX[nTrain:]
nTrain,nFeats = trainX.shape
indexes = np.arange(0,nFeats,1)
print nTrain,nFeats

clf = lm.Ridge(alpha=146)
Cs = np.logspace(6,10,10,base=2)
for c in Cs:
	clf.alpha = c
	t = time()
	auc = cv(clf,trainX,Y,proba=False)
	print 'C',c,'auc',auc,'time',time()-t

clf = lm.LogisticRegression(C=16.)
clf.fit(trainX, Y)
preds = clf.predict_proba(testX)[:, 1]
filename = raw_input("Enter name for submission file: ")
save_results(preds, filename + ".csv")