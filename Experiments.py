# -*- coding: utf-8 -*-

#### Script for Liwc vocabulary evaluation
####
#### There will be used three dictionaries:
####
####  LIWC for Portuguese (Subject of the evaluation)
####  OpinionLexicon
####  SentiLex
####
#### Two experiments are conducted
####          Vocabulary Agreeement
####          Score eachieved by each vocabulary in a classification task
####
#### The script runs on python 2.7. All the information is printed on the screen
####
####


#### Author: Pedro Paulo Balage Filho
#### Version: 1.0
#### Date: 05/12/12


# Import Libraries
# Necessary NLTK library
# Each dictionary has its own reader
from ReLi import ReLiCorpusReader
from Liwc import LiwcReader
from OpinionLexicon import OpLexiconReader
from SentiLex import SentiLexReader
from LexiconClassifier import Classifier
from nltk.metrics import ConfusionMatrix, precision, recall, f_measure, accuracy

# Load Dictionaries
liwc = LiwcReader('Dictionaries/LIWC/LIWC2007_Portugues_win.dic')
oplexicon = OpLexiconReader('Dictionaries/oplexicon/lexico_v2.1txt')
sentilex =  SentiLexReader('Dictionaries/SentiLex/SentiLex-flex-PT02.txt')

##############################################
######  Vocabulary Disagreement
##############################################

# Simple agreement. Counts all words: neutral, positive and negative
print '########## Agreement ###########'

dictionaries = [liwc,oplexicon,sentilex]
agreement = dict()

# all dictionaries must have the methods vocabulary(), polarity() and get_name()
for i,dict_i in enumerate(dictionaries):
    for j,dict_j in enumerate(dictionaries):
        # avoid to do check between the same dictionaries again
        if i < j:
            vocab_i = dict_i.vocabulary()
            vocab_j = dict_j.vocabulary()
            agree = 0
            disagree = 0
            # I am only computing the vocabulary both have in common
            # (intersection)
            same_vocab = vocab_i.intersection(vocab_j)
            total = len(same_vocab)
            for element in list(same_vocab):
                if dict_i.polarity(element) == dict_j.polarity(element):
                    agree += 1
                else:
                    disagree += 1
            name_i = dict_i.get_name()
            name_j = dict_j.get_name()
            if name_i not in agreement:
                agreement[name_i] = dict()
            agreement[name_i][name_j] = '{:.2%} out of {} entries'.format(float(agree) / float(total),total)

# print agreement information
for i in agreement:
    for j in agreement[i]:
            print 'Agreement between ',i,' and ', j, ' :', agreement[i][j]

# Polar agreement. Only uses positive or negative terms
print '########### Polar Agreement #############'

dictionaries = [liwc,oplexicon,sentilex]
agreement = dict()

# all dictionaries must have the methods vocabulary_polar(), polarity() and get_name()
for i,dict_i in enumerate(dictionaries):
    for j,dict_j in enumerate(dictionaries):
        if i < j:
            vocab_i = dict_i.vocabulary_polar()
            vocab_j = dict_j.vocabulary_polar()
            agree = 0
            disagree = 0
            # intersection of polar vocabulary
            same_vocab = vocab_i.intersection(vocab_j)
            total = len(same_vocab)
            for element in list(same_vocab):
                if dict_i.polarity(element) == dict_j.polarity(element):
                    agree += 1
                else:
                    disagree += 1
            name_i = dict_i.get_name()
            name_j = dict_j.get_name()
            if name_i not in agreement:
                agreement[name_i] = dict()
            agreement[name_i][name_j] = '{:.2%} out of {} entries'.format(float(agree) / float(total),total)

for i in agreement:
    for j in agreement[i]:
            print 'Agreement between ',i,' and ', j, ' :', agreement[i][j]


##### Some examples of LIWC agreements and disagreements with opinion lexicon #########

liwc_vocab = liwc.vocabulary_polar()
oplexicon_vocab = oplexicon.vocabulary_polar()
sentilex_vocab = sentilex.vocabulary_polar()

agree = []
disagree = []
same_vocab = liwc_vocab.intersection(oplexicon_vocab)
total = len(same_vocab)

for element in list(same_vocab):
    # check agreement
    if liwc.polarity(element) == oplexicon.polarity(element):
        agree.append((element,liwc.polarity(element)))
    else:
        disagree.append((element,liwc.polarity(element),oplexicon.polarity(element)))

# print the examples
print '\n\nAGREEMENT EXAMPLES (OpinionLexicon)'
print 'pol\tword'

for word, pol in sorted(agree[:20]):
    print pol,'\t', word

print '\n\nDISAGREEMENT EXAMPLES (OpinionLexicon)'
print 'lwic\toplex\tword'
for word, pol1, pol2 in sorted(disagree[:40]):
    print pol1,'\t', pol2, '\t'  , word



##############################################
######  Performance in classification
##############################################

reli = ReLiCorpusReader()

#################### Predicate classification ######################

# In a aspect-based sentiment analysis, the opinion
# has the aspect (or feature) and the predicate (evaluation over the aspect)
negative_aspects = reli.opinion_aspects(polarity='negative')
negative_words = []
for aspect,predicate,pol in negative_aspects:
        negative_words.append(predicate)


positive_aspects = reli.opinion_aspects(polarity='positive')
positive_words = []
for aspect,predicate,pol in positive_aspects:
        positive_words.append(predicate)

print '#########################################################################'
print '########################  Opinion  classification #######################'
print '#########################################################################'

dictionaries = [liwc,oplexicon,sentilex]


for dictionary in dictionaries:

    # from LexiconClassifier library
    classifier = Classifier(dictionary)

    # build the train and test set
    word_vector = negative_words + positive_words
    gold_standard = [-1 for i in range(len(negative_words))] + [1 for i in range(len(positive_words))]
    results = [classifier.classify(s) for s in word_vector]

    # print the classification results
    print 'Dictionary : ', dictionary.get_name(), '\n'
    print ConfusionMatrix(gold_standard,results).pp()
    print 'Accuracy: ', accuracy(gold_standard,results)
    for c in [0,1,-1]:
        print 'Metrics for class ', c
        gold = set()
        test = set()
        for i,x in enumerate(gold_standard):
            if x == c:
                gold.add(i)
        for i,x in enumerate(results):
            if x == c:
                test.add(i)
        print 'Precision: ', precision(gold, test)
        print 'Recall   : ', recall(gold, test)
        print 'F_measure: ', f_measure(gold, test)
    print '\n\n'


#################### Sentences classification ##########################

# Not reported in the paper because LIWC doesn't have neutral class

positive_sents = [reli.words_sentence_pos(s) for s in reli.sents(polarity='positive')]
negative_sents = [reli.words_sentence_pos(s) for s in reli.sents(polarity='negative')]
neutral_sents = [reli.words_sentence_pos(s) for s in reli.sents(polarity='neutral')]


print '#########################################################################'
print '###################### Sentences classification #########################'
print '#########################################################################'
dictionaries = [liwc,oplexicon,sentilex]

for dictionary in dictionaries:
    classifier = Classifier(dictionary)
    sentence_vector = negative_sents + positive_sents + neutral_sents
    sitive_sents = [reli.words_sentence_pos(s) for s in reli.sents(polarity='positive')]
    gold_standard = [-1 for i in range(len(negative_sents))]
    gold_standard += [1 for i in range(len(positive_sents))] + [0 for i in range(len(neutral_sents))]
    results = [classifier.classify(s) for s in sentence_vector]
    print 'Dictionary : ', dictionary.get_name(), '\n'
    print ConfusionMatrix(gold_standard,results).pp()
    print 'Accuracy: ', accuracy(gold_standard,results)
    for c in [0,1,-1]:
        print 'Metrics for class ', c
        gold = set()
        test = set()
        for i,x in enumerate(gold_standard):
            if x == c:
                gold.add(i)
        for i,x in enumerate(results):
            if x == c:
                test.add(i)
        print 'Precision: ', precision(gold, test)
        print 'Recall   : ', recall(gold, test)
        print 'F_measure: ', f_measure(gold, test)
    print '\n\n'

#################### Polar sentence classification ##########################

print '#########################################################################'
print '################### Polar Sentences classification ######################'
print '#########################################################################'
dictionaries = [liwc,oplexicon,sentilex]

for dictionary in dictionaries:
    classifier = Classifier(dictionary)
    sentence_vector = negative_sents + positive_sents
    gold_standard = [-1 for i in range(len(negative_sents))] + [1 for i in range(len(positive_sents))]
    results = [classifier.classify(s) for s in sentence_vector]
    print 'Dictionary : ', dictionary.get_name(), '\n'
    print ConfusionMatrix(gold_standard,results).pp()
    print 'Accuracy: ', accuracy(gold_standard,results)
    for c in [0,1,-1]:
        print 'Metrics for class ', c
        gold = set()
        test = set()
        for i,x in enumerate(gold_standard):
            if x == c:
                gold.add(i)
        for i,x in enumerate(results):
            if x == c:
                test.add(i)
        print 'Precision: ', precision(gold, test)
        print 'Recall   : ', recall(gold, test)
        print 'F_measure: ', f_measure(gold, test)
    print '\n\n'

