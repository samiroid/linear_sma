import codecs
from ipdb import set_trace
import numpy as np
#Preprocess NRC lexicons

#S140
NRC_140_LEXICONS = "/Users/samir/Dev/resources/lexicons/Sentiment140/"
NRC_HASHTAG_LEXICONS = "/Users/samir/Dev/resources/lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/"

DUYU_LEX = "/Users/samir/Dev/projects/xpandlex/DATA/txt/Duyu-Lex/lexicon-coling2014/"

def read_semeval(path):

    with open(path) as fid: 
        lex =  { wrd: float(scr) for wrd, scr in (line.split('\t') for line in fid) }
    lex = normalize_scores(lex)
    return lex

def read_duyu():
    lex = {}
    for f in ["pos.txt","neg.txt"]:
        with open(DUYU_LEX+f) as fid:
            for l in fid:
                spt = l.split()
                if len(spt) > 2:
                    continue
                else:
                    lex[spt[0]] = round(float(spt[1]),3)
    return lex

def read_NRC_hashtag_lex(norm_scores=False):

    unigram_lexicon = read_nrc_aux(NRC_HASHTAG_LEXICONS, norm_scores)
    return unigram_lexicon

def read_sentiment140_lex(norm_scores=False):
    unigram_lexicon = read_nrc_aux(NRC_140_LEXICONS, norm_scores)
    return unigram_lexicon

def read_nrc_aux(path, norm_scores=False):
    unigram_lexicon = {}       
    unigram_file = open(path+'unigrams-pmilexicon.txt')
    scores = []
    for line in unigram_file:
        word, pmi_score, num_positive, num_negative = line.strip().split()
        pmi_score = float(pmi_score)        
        scores.append(pmi_score)
        unigram_lexicon[word] = pmi_score
    unigram_file.close()
    if norm_scores:
        unigram_lexicon = normalize_scores(unigram_lexicon)
    return unigram_lexicon


def normalize_scores(lexicon, to_range=(-1,1)):
    scores = lexicon.values()
    old_range = (min(scores),max(scores))
    for k in lexicon.keys():
        lexicon[k] = linear_conversion(old_range,to_range,lexicon[k])

    return lexicon

def linear_conversion(source_range, dest_range, val):
    MIN = 0
    MAX = 1
    val = float(val)
    source_range = np.asarray(source_range,dtype=float)
    dest_range = np.asarray(dest_range,dtype=float)
    #new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    new_value = ( (val - source_range[MIN]) / (source_range[MAX] - source_range[MIN]) ) *\
                (dest_range[MAX] - dest_range[MIN]) + dest_range[MIN]
    return round(new_value,3)


semeval_lex="DATA/input/SemEval2015_taskE_gold.txt"
sem_lex = read_semeval(semeval_lex)
ord_sem_lex = sorted(sem_lex.items(),key=lambda x:x[1],reverse=True)

with codecs.open("DATA/input/sem_lex.txt","w") as fid:
    for w,l in ord_sem_lex:
        fid.write("%s\t%.3f\n" % (w,l))

semeval_trial="DATA/input/SemEval2015_taskE_trial.txt"
sem_lex_trial = read_semeval(semeval_trial)
ord_sem_lex_trial = sorted(sem_lex_trial.items(),key=lambda x:x[1],reverse=True)

with codecs.open("DATA/input/sem_lex_trial.txt","w") as fid:
    for w,l in ord_sem_lex_trial:
        fid.write("%s\t%.3f\n" % (w,l))

duyu_lex = read_duyu()
ord_duyu_lex = sorted(duyu_lex.items(),key=lambda x:x[1],reverse=True)

with codecs.open("DATA/input/duyu_lex.txt","w") as fid:
    for w,l in ord_duyu_lex:
        fid.write("%s\t%.3f\n" % (w,l))

import sys; sys.exit()

s140 = read_sentiment140_lex(norm_scores=True)
ord_s140 = sorted(s140.items(),key=lambda x:x[1],reverse=True)

with codecs.open("DATA/input/s140.txt","w") as fid:
	for w,l in ord_s140:
		fid.write("%s\t%.3f\n" % (w,l))


ht_lex = read_NRC_hashtag_lex(norm_scores=True)
ord_ht_lex = sorted(ht_lex.items(),key=lambda x:x[1],reverse=True)

with codecs.open("DATA/input/ht_lex.txt","w") as fid:
	for w,l in ord_ht_lex:
		fid.write("%s\t%.3f\n" % (w,l))
