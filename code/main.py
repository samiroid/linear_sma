import argparse
import cPickle
from collections import Counter
from ipdb import set_trace
import numpy as np
import os
from scipy.sparse import hstack
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import sma_toolkit as st

def get_parser():
    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument('-tr', type=str, help='train file')
    parser.add_argument('-ts', type=str, nargs='+', help='test file(s)')
    parser.add_argument('-cv', type=int, help='n. of cross-validation folds')
    parser.add_argument('-res',type=str, required=True, help="path to the results")
    parser.add_argument('-bow', type=str, choices=['bin','freq'], help='bow weights: bin:binary|freq:frequency')
    parser.add_argument('-boe', type=str, help='path to word embeddings. If set, use a bag-of-embeddings representations (summing embeddings)')    
    parser.add_argument('-mean_emb', action="store_true", help='mean embeddings')   
    
    return parser

def bow_features(X_idx,bow_weights):
	X = np.zeros((len(X_idx),len(wrd2idx)))			
	if bow_weights == "bin":
		for i, doc in enumerate(X_idx):	
			if len(doc)>0 : X[i,doc] = 1  			
	elif bow_weights == "freq":
		for i, doc in enumerate(X_idx):	
			if len(doc)>0 : 
				#word counts
				ctr = Counter(doc)						
				X[i,doc] = [ctr[w] for w in doc]
	else:
		raise NotImplementedError
	return X

def boe_features(X_idx, E, mean=False):	
	# BOE features
	X = np.zeros((len(X_idx),E.shape[0]))		
	for i, doc in enumerate(X_idx):						
		if mean: X[i,:] = E[:,doc].T.mean(axis=0)	
		else:    X[i,:] = E[:,doc].T.sum(axis=0)			
	return X

if __name__=="__main__":	

	parser = get_parser()
	args = parser.parse_args()
	
	assert (args.ts is not None and args.cv is None) or \
		   (args.cv is not None and args.ts is None)	
	
	print "[Training Data: %s" % os.path.basename(args.tr)
	datasets=[]
	feature_set = set()	
	print "[building feature vectors]"
	for dataset in [args.tr] + args.ts:		
		print "\t> %s" % dataset
		with open(dataset) as fid:		
			wrd2idx, X_idx, Y = cPickle.load(fid)		
			Y = np.array(Y)
		X = None
		# BOW features
		if args.bow is not None:
			feature_set.add("BOW_"+args.bow)				
			X = bow_features(X_idx, args.bow)	
		# BOE features	
		if args.boe is not None:
			if args.mean_emb: 
				feature_set.add("BOE_mean")		
			else:
				feature_set.add("BOE")
			#load pre-trained embeddings
			with open(args.boe) as fid: E = cPickle.load(fid)
			word_vectors = boe_features(X_idx, E, args.mean_emb)
			if X is None: 
				X = word_vectors
			else:         
				X = np.hstack((X ,word_vectors))	
		assert X is not None, "No features were extracted!"
		datasets.append([dataset, X,Y])	
	features = "+".join(feature_set)
	print "[feature set: %s] " % features
	if not os.path.exists(args.res): os.makedirs(args.res)        	

	if args.cv is None:	
		_,X,Y = datasets[0]
		clf = LogisticRegression()		
		clf.fit(X,Y)
		for test_set, test_X, test_Y in datasets[1:]:						
			Y_hat = clf.predict(test_X)
			prec  = precision_score(test_Y, Y_hat, average="macro")
			acc   = accuracy_score(test_Y, Y_hat)			
			print "   %s > acc: %.3f | avg prec: %.3f" % (os.path.basename(test_set), acc, prec)	
			with open("%s%s-%s" % (args.res, os.path.basename(args.tr),features),"w") as fod:
				fod.write("%s,%.3f,%.3f" % (features, acc, prec))
	elif args.cv:
		print "[cross-validation: %d folds]" % args.cv			
		accs  = []
		precs = []
		for i, fold in enumerate(st.kfolds(args.cv, len(Y),shuffle=True)):
			train, test = fold
			# set_trace()
			X_train = X[train,:]
			Y_train = Y[train]
			X_test  = X[test,:]
			Y_test  = Y[test]					
			clf = LogisticRegression()
			clf.fit(X_train,Y_train)	
			y_hat = clf.predict(X_test)
			prec  = precision_score(Y_test, y_hat)
			acc   = accuracy_score(Y_test, y_hat)	
			accs  += [acc]			
			precs += [prec]
			print "%s (%d) ** acc: %.3f | avg prec: %.3f" % (features, i, acc, prec)			
		print ""		
		print "%s ** acc: %.3f (%.3f) | avg prec: %.3f (%.3f)" % (features, np.mean(accs), np.std(accs), np.mean(precs), np.std(precs))
		fname=os.path.basename(os.path.splitext(args.tr)[0])
		with open("DATA/res/%s-%d-%s" % (fname,args.cv,features),"w") as fod:
			fod.write("%s,%s,%.3f,%.3f,%.3f,%.3f" % (fname, features, np.mean(accs), np.mean(precs), np.std(accs), np.std(precs)))
	else:
		raise NotImplementedError, "sorry but I don't know what to do :("
