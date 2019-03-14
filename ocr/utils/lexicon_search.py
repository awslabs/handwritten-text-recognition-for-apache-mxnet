# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import nltk
import numpy as np
from nltk.metrics import *
from nltk.util import ngrams
import enchant  # spell checker library pyenchant
from enchant.checker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.corpus import words
import string
import re
from collections import Counter
from nltk.corpus import brown

from nltk.probability import FreqDist
from nltk.metrics import edit_distance

from sympound import sympound
import re
from weighted_levenshtein import lev
    
class WordSuggestor():
    '''
    Code obtained from http://norvig.com/spell-correct.html.
    '''
    def __init__(self):
        self.words = Counter(brown.words())
    
    def P(self, word): 
        "Probability of `word`."
        N = sum(self.words.values())
        return self.words[word] / N

    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(candidates(word), key=self.P)

    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.words)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

class OcrDistanceMeasure():
    # Helper class to obtain a handwriting error weighted edit distance. The weighted edit distance class can be found in
    # https://github.com/infoscout/weighted-levenshtein.
    # Substitute_costs.txt, insertion_costs and deletion_costs are calculated in
    # https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC/blob/language_model/model_distance.ipynb
    
    def __init__(self):
        self.substitute_costs = self.make_substitute_costs()
        self.insertion_costs = self.make_insertion_costs()
        self.deletion_costs = self.make_deletion_costs()

    def make_substitute_costs(self):
        substitute_costs = np.loadtxt('models/substitute_costs.txt', dtype=float)
        #substitute_costs = np.ones((128, 128), dtype=np.float64)
        return substitute_costs
    
    def make_insertion_costs(self):
        insertion_costs = np.loadtxt('models/insertion_costs.txt', dtype=float)
        #insertion_costs = np.ones(128, dtype=np.float64)
        return insertion_costs
    
    def make_deletion_costs(self):
        deletion_costs = np.loadtxt('models/deletion_costs.txt', dtype=float)
        #deletion_costs = np.ones(128, dtype=np.float64)
        return deletion_costs
    
    def __call__(self, input1, input2):
        return lev(input1, input2, substitute_costs=self.substitute_costs,
                  insert_costs=self.insertion_costs,
                  delete_costs=self.deletion_costs)
    
class LexiconSearch:
    '''
    Lexicon search was based on https://github.com/rameshjesswani/Semantic-Textual-Similarity/blob/master/nlp_basics/nltk/string_similarity.ipynb
    '''
    def __init__(self):
        self.dictionary = enchant.Dict('en')
        self.word_suggestor = WordSuggestor()
        self.distance_measure = OcrDistanceMeasure()
                
    def suggest_words(self, word):
        candidates = list(self.word_suggestor.candidates(word))
        output = []
        for word in candidates:
            if word[0].isupper():
                output.append(word)
            else:
                if self.dictionary.check(word):
                    output.append(word)
        return output
            
    def minimumEditDistance_spell_corrector(self,word):
        max_distance = 3

        if (self.dictionary.check(word.lower())):
            return word

        suggested_words = self.suggest_words(word)
        num_modified_characters = []
        
        if len(suggested_words) != 0:
            for sug_words in suggested_words:
                num_modified_characters.append(self.distance_measure(word, sug_words))
                
            minimum_edit_distance = min(num_modified_characters)
            best_arg = num_modified_characters.index(minimum_edit_distance)
            if max_distance > minimum_edit_distance:
                best_suggestion = suggested_words[best_arg]
                return best_suggestion
            else:
                return word
        else:
            return word
