import argparse
import cPickle
from collections import Counter
from pdb import set_trace
import numpy as np
import os
from scipy.sparse import hstack
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import sma_toolkit as st
from sma_toolkit.evaluation import FmesSemEval

def get_parser():
	parser = argparse.ArgumentParser(description="Classifier")
	parser.add_argument('-tr', type=str, help='train file')
	parser.add_argument('-ts', type=str, nargs='+', help='test file(s)')
	parser.add_argument('-dev', type=str, help='dev file')
	parser.add_argument('-cv', type=int, help='n. of cross-validation folds')
	parser.add_argument('-res',type=str, required=True, help="path to the results")
	parser.add_argument('-res_append', action="store_true", help="append the results")
	parser.add_argument('-polar_f1', type=str, nargs=2, help="positive and negative classes to calculate 'polar F1'. Otherwise avg F1 will be reported")
	parser.add_argument('-bow', type=str, choices=['bin','freq'], help='bow weights: bin:binary|freq:frequency')
	parser.add_argument('-boe', type=str, help='path to word embeddings. If set, use a bag-of-embeddings representations (summing embeddings)')    
	parser.add_argument('-boe_mean', type=str, help='path to word embeddings. If set, use a bag-of-embeddings representations (mean embeddings)')    
    
    
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
			wrd2idx, lbl2idx, X_idx, Y = cPickle.load(fid)		
			Y = np.array(Y)
		X = None
		# BOW features
		if args.bow is not None:
			feature_set.add("BOW_"+args.bow)				
			X = bow_features(X_idx, args.bow)	
		# BOE features	
		if args.boe is not None:						
			feature_set.add("BOE")				
			#load pre-trained embeddings
			with open(args.boe) as fid: E = cPickle.load(fid)
			word_vectors = boe_features(X_idx, E, False)
			if X is None: 
				X = word_vectors
			else:         
				X = np.hstack((X ,word_vectors))	
		if args.boe_mean is not None:						
			feature_set.add("BOE_mean")				
			#load pre-trained embeddings
			with open(args.boe) as fid: E = cPickle.load(fid)
			word_vectors = boe_features(X_idx, E, True)
			if X is None: 
				X = word_vectors
			else:         
				X = np.hstack((X ,word_vectors))	
		assert X is not None, "No features were extracted!"
		datasets.append([dataset, X,Y])	
	features = "+".join(feature_set)
	print "[feature set: %s] " % features

	res_folder = os.path.dirname(args.res)
	if not os.path.exists(res_folder): os.makedirs(res_folder)        	
	train_fname  = os.path.basename(os.path.splitext(args.tr)[0])
	if args.cv is None:	
		if args.res_append:        
			fod = open(args.res,"a")        
		else:        
			fod = open(args.res,"w")
			if args.polar_f1 is not None:
				fod.write("model, train_set, test_set, acc, polar_f1\n")
			else:
				fod.write("model, train_set, test_set, acc, fm\n")    	
		_,X,Y = datasets[0]
		clf = LogisticRegression()		
		clf.fit(X,Y)
		for test_set, test_X, test_Y in datasets[1:]:				
			Y_hat = clf.predict(test_X)
			acc   = accuracy_score(test_Y, Y_hat)					
			fname  = os.path.basename(os.path.splitext(test_set)[0])
			if args.polar_f1 is not None:
				f1 = FmesSemEval(Y_hat, test_Y, 
								 lbl2idx[args.polar_f1[1]], 
								 lbl2idx[args.polar_f1[0]])
				print "   %s > acc: %.3f | polar f1: %.3f" % (fname, acc, f1)	
			else:
				f1 = f1_score(test_Y, Y_hat, average="macro")
				print "   %s > acc: %.3f | avg f1: %.3f" % (fname, acc, f1)		
			
			fod.write("%s,%s,%s,%.3f,%.3f\n" % (features, train_fname, fname, acc, f1))			
		fod.close()

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
			prec  = f1_score(Y_test, y_hat)
			acc   = accuracy_score(Y_test, y_hat)	
			accs  += [acc]			
			precs += [prec]
			print "%s (%d) ** acc: %.3f | avg prec: %.3f" % (features, i, acc, prec)			
		print ""		
		print "%s ** acc: %.3f (%.3f) | avg prec: %.3f (%.3f)" % (features, np.mean(accs), np.std(accs), np.mean(precs), np.std(precs))
		raise NotImplementedError, "logging"		
	else:
		raise NotImplementedError, "sorry but I don't know what to do :("
