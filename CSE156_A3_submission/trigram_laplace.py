#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys
from tqdm import tqdm

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

class Trigram(LangModel):
    def __init__(self,unigram, laplace = 1):
        self.model = dict()
        self.bigram = dict()
        self.laplace = laplace
        self.unique = unigram.vocab()

    def inc_word(self, w, model):
        if w in model:
            model[w] += 1.0
        else:
            model[w] = 1.0

    def fit_sentence(self, sentence):
        sentence = ['*', '**'] + sentence
        sentence.append('END_OF_SENTENCE')
        trigram = [(x) for x in list(zip(sentence[:-1], sentence[1:], sentence[2:]))]
        bigram = [(x) for x in list(zip(sentence[:-1], sentence[1:]))]
        for w in trigram:
            self.inc_word(w, self.model)
        for w in bigram:
            self.inc_word(w, self.bigram)

    def norm(self):
        """Normalize and convert to log2-probs."""
        # tot = 0.0
        # for word in self.model:
        #     tot += self.model[word]
        # ltot = log(tot, 2)
        # for word in self.model:
        #     biwords = (word[0], word[1])
        #     self.model[word] = log(((self.model[word] + self.laplace)/(self.bigram[biwords] + self.laplace * len(self.unique))),2)

        for x in tqdm(self.model):
            self.model[x] = log(((self.model[x] + self.laplace) / (self.bigram[(x[0],x[1])] + self.laplace * len(self.unique)) ), 2)
 
    # def cond_logprob(self, word, previous):
    #     tri_words = (previous[0], previous[1], word)
    #     bi_words = (previous[0], previous[1])
    #     if tri_words in self.model:
    #         return self.model[tri_words]
    #     else:
    #         if bi_words in self.bigram:
    #             return log((self.laplace/self.bigram[bi_words] + self.laplace*len(self.unique)),2)
    #         else:
    #             return log((self.laplace/(self.laplace * len(self.unique))),2)

    def cond_logprob(self, word, previous):
        tri_words = (previous[0], previous[1], word)
        if previous in self.bigram:
            if tri_words in self.model:
                return self.model[tri_words]
            else:
                prob = self.laplace / (self.bigram[previous] + self.laplace * len(self.unique))
                return log(prob, 2)
        else:
            prob = self.laplace / (self.laplace*len(self.unique))
            return log(prob, 2)

    def vocab(self):
        return self.model.keys()

    # def logprob_sentence(self, sentence):
    #     p = 0.0
    #     sentence = ['SOS', 'SOS'] + sentence
    #     sentence.append('END_OF_SENTENCE')
    #     for i in xrange(2, len(sentence)):
    #         p += self.cond_logprob(sentence[i], sentence[:i])
    #     p += self.cond_logprob('END_OF_SENTENCE', sentence)
    #     return p

    def logprob_sentence(self, sentence):
        p = 0.0
        sentence = ['*', '**'] + sentence
        sentence.append('END_OF_SENTENCE')
        for i in range(len(sentence)-2):
            p += self.cond_logprob(sentence[i+2], (sentence[i],sentence[i+1]))
        return p

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        print("start finding entropy")
        for s in tqdm(corpus):
            num_words += len(s) + 3 # for EOS and 2 SOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)













