import argparse
from ipdb import set_trace
import my_utils as ut
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import codecs
import cPickle
import sys
from collections import defaultdict

def load_lexicon(path, sep='\t', lex_thresh=0.2):
	#we might want to filter out "neutral words"	
	with open(path) as fid:	
		lex =  { wrd: float(scr) for wrd, scr in (line.split('\t') for line in fid) if float(scr) < -lex_thresh or float(scr) > lex_thresh }
	return lex

def get_parser():
    parser = argparse.ArgumentParser(description="Lexicon Classifier")
    
    #Basic Input
    parser.add_argument('-lex', type=str, required=True,
                        help='lexicon file')

    parser.add_argument('-class_threshold', type=float, default=0.1,
                        help='decicions threshold')

    parser.add_argument('-neutral_threshold', type=float,
                        help='threshold for neutral class')

    parser.add_argument('-ts', type=str, required=True, nargs='+',
                        help='test file(s)')

    parser.add_argument('-res', type=str, required=True,
                        help='results file')

    return parser

if __name__=="__main__":	
	parser = get_parser()
	args = parser.parse_args()  
	lex = load_lexicon(args.lex)
	print "testing"
	try:
		with open(args.res) as fid:
			res = cPickle.load(fid)
	except IOError:
		res = defaultdict(dict)

	#evaluate 
	for test_file in args.ts:				
		labels_test = []
		scores = []
		with open(test_file) as fid:
			for l in fid:
				splt = l.split("\t")
				labels_test.append(splt[0])		
				word_scores = map(lambda x:lex[x] if x in lex else 0, splt[1].split())
				#remove zeroes
				word_scores = filter(lambda x:x!=0, word_scores)
				if len(word_scores)>0:
					msg_score = np.mean(word_scores)				
				else:				
					msg_score = 0		
				scores.append(np.mean(msg_score))
		#map labels to numeric values
		lbl2idx = ut.word_2_idx(labels_test)
		Y_test  = np.array([lbl2idx[l] for l in labels_test])
		#classification
		if args.neutral_threshold:
			Y_hat = map(lambda x: lbl2idx["positive"] if x >= args.class_threshold+args.neutral_threshold else \
				               	  lbl2idx["negative"] if x <= args.class_threshold-args.neutral_threshold else \
				                  lbl2idx["neutral"], scores)
			avgF1 = f1_score(Y_test, Y_hat,average="macro") 		
		else:
			Y_hat = map(lambda x:lbl2idx["positive"] if x >= args.class_threshold else lbl2idx["negative"], scores) 
			avgF1 = f1_score(Y_test, Y_hat,average="binary")
			
		acc = accuracy_score(Y_test, Y_hat)				
		print "%s ** acc: %.3f | F1: %.3f " % (test_file, acc, avgF1)
		res[args.lex][test_file] = round(avgF1,3)

	with open(args.res,"w") as fid:
		cPickle.dump(res, fid)