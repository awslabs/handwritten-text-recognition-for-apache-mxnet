import nltk
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

# Source: https://github.com/rameshjesswani/Semantic-Textual-Similarity/blob/master/nlp_basics/nltk/string_similarity.ipynb
# Source: http://norvig.com/spell-correct.html

class WordSuggestor():
    def __init__(self):
        self.words = Counter(brown.words())
    
    # def words(self, text):
    #     return re.findall(r'\w+', text.lower())

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
    
class LexiconSearch:
    def __init__(self):
        self.dictionary = enchant.Dict('en')
        self.word_suggestor = WordSuggestor()
        self.stemmer = PorterStemmer()
        
    def suggest_words(self, word):
        return list(self.word_suggestor.candidates(word))
    
    def levenshtein_distance(self,s1,s2):
        # nltk already have implemeted function
        distance_btw_strings = edit_distance(s1,s2)
        return distance_btw_strings
    
    def ngram(self,word,n):
        grams = list(ngrams(word,n))
        return grams
        
    def check_mistakes_in_sentence(self,sentence):
        misspelled_words = []
        self.check.set_text(sentence)
        
        for err in self.check:
            misspelled_words.append(err.word)
            
        if len(misspelled_words) == 0:
            print(" No mistakes found")
        return misspelled_words
    
    def jaccard(self,a,b):

        union = list(set(a+b))
        intersection = list(set(a) - (set(a)-set(b)))
        jaccard_coeff = float(len(intersection))/len(union)
        return jaccard_coeff

    def minimumEditDistance_spell_corrector(self,word):
        
        max_distance = 2

        if (self.dictionary.check(word)):
            return word

        suggested_words = self.suggest_words(word)
        num_modified_characters = []
        
        if suggested_words != 0:
            
            for sug_words in suggested_words:
                num_modified_characters.append(self.levenshtein_distance(word,sug_words))
                
            minimum_edit_distance = min(num_modified_characters)
            best_arg = num_modified_characters.index(minimum_edit_distance)
            if max_distance > minimum_edit_distance:
                best_suggestion = suggested_words[best_arg]
                return best_suggestion
            else:
                return word
        else:
            return word
        
    def ngram_spell_corrector(self, word):
        exclude = set(string.punctuation)
        word_without_punctuation = ''.join(ch for ch in word if ch not in exclude)

        max_distance = 2
        if word in string.punctuation:
            return word
        
        if (self.dictionary.check(word)):
            return word
        suggested_words = self.suggest_words(word)
        
        num_modified_characters = []
       
        max_jaccard = []
        list_of_sug_words = []
        if suggested_words != 0:
            
            word_ngrams = self.ngram(word,2)

            for sug_words in suggested_words:

                if (self.levenshtein_distance(word,sug_words)) < 3 :

                    sug_ngrams = self.ngram(sug_words,2)
                    jac = self.jaccard(word_ngrams,sug_ngrams)
                    max_jaccard.append(jac)
                    list_of_sug_words.append(sug_words)
            highest_jaccard = max(max_jaccard)
            best_arg = max_jaccard.index(highest_jaccard)
            word = list_of_sug_words[best_arg]
            return word
        else:
            return word
