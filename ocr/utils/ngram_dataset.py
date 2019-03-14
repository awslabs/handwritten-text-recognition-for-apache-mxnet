# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pickle
from mxnet.gluon.data import dataset

class Ngram_dataset(dataset.ArrayDataset):
    '''
    The ngram_dataset takes a noisy_forms_dataset object and converts it into an ngram dataset.
    That is, for each word/characters i a tuple is created that contains:
       * n words/characters before
       * n words/characters after
       * Noisy word/characters i
       * The actual word/character i

    Parameters
    ----------
    passage_ds: Noisy_forms_dataset
        A noisy_forms_dataset object

    name: string
        An identifier for the temporary save file.

    output_type: string, options: ["word", "character"]
        Output data in terms of n characters of words

    n: int, default 3
        The number of characters to output
    '''
    def __init__(self, passage_ds, name, output_type, n=3):
        self.passage_ds = passage_ds
        self.n = n
        
        output_type_options = ["word", "character"]
        assert output_type in output_type_options, "{} is not a valid type. Available options: {}".format(
            output_type, outptu_type_options)
        self.output_type = output_type
        
        root = os.path.join("dataset", "noisy_forms")
        if not os.path.isdir(root):
            os.makedirs(root)
        ds_location = os.path.join(root, "ngram_{}.pickle".format(name))
        if os.path.isfile(ds_location):
            data = pickle.load(open(ds_location, 'rb'))
        else:
            data = self._get_data()
            pickle.dump(data, open(ds_location, 'wb'), protocol=2)
        super(Ngram_dataset, self).__init__(data)

    def _get_n_grams(self, text_arr, idx, pre, n):
        output = []
        if pre:
            indexes = range(idx - n, idx)
        else:
            indexes = range(idx + 1, idx + n + 1)
        for i in indexes:
            if 0 <= i and i < len(text_arr):
                word = text_arr[i]
                if len(word) == 0:
                    output.append(0)
                else:
                    output.append(word)
            else:
                output.append(0)
        return output

    def _remove_empty_words(self, text_arr):
        # Helper function to remove empty words
        output = []
        for word in text_arr:
            if len(word) > 0:
                output.append(word)
        return output

    def _separate_word_breaks(self, text_arr):
        # Helper function to separate words that are split into two with "-"
        output = []

        for word in text_arr:
            if "-" in word:
                words = word.split("-")
                for word_i in words:
                    if len(word) > 0:
                        output.append(word_i)
            else:
                output.append(word)    
        return output

    def _is_ngram_similar(self, ngram1, ngram2, p1=0.8, p2=0.8):
        '''
        Helper function to check if ngram1 is similar to ngram2.
        Parameters
        ----------
        ngram1: [str]
            A list of strings (or 0 for the null character) that is of size n.
        ngram2: [str]
            A list of strings (or 0 for the null character) that is of size n.

        p1: float
            The percentage of characters that are the same within 2 words to be considered the same word.

        p2: float
            The percentage of words that are the same for ngram1 and ngram2 to be considered similar

        Return
        ------
        is_ngram_similar: bool
            Boolearn that indicates ngram1 and ngram2 are similar.
        ''' 

        in_count = []
        for ngram1_i, ngram2_i in zip(ngram1, ngram2):
            if ngram1_i == 0 or ngram2_i == 0:
                if ngram1_i == ngram2_i:
                    in_count.append(1)
                else:
                    in_count.append(0)
            else:
                ngram1_i_np = np.array(list(ngram1_i))
                ngram2_i_np = np.array(list(ngram2_i))
                if np.mean(np.in1d(ngram1_i_np, ngram2_i_np)) > p1:
                    in_count.append(1)
                else:
                    in_count.append(0)
        is_ngram_similar = np.mean(in_count) > p2
        return is_ngram_similar
            
    def _get_data(self):
        ngrams = []
        for i in range(len(self.passage_ds)):
            noisy_text, text = self.passage_ds[i]
            noisy_text_arr, text_arr = noisy_text.split(" "), text.split(" ")

            # Heuristics
            noisy_text_arr = self._separate_word_breaks(noisy_text_arr)
            text_arr = self._separate_word_breaks(text_arr)
            
            noisy_text_arr = self._remove_empty_words(noisy_text_arr)
            text_arr = self._remove_empty_words(text_arr)

            for j in range(len(noisy_text_arr)):
                pre_values_j = self._get_n_grams(noisy_text_arr, j, pre=True, n=3)
                post_values_j = self._get_n_grams(noisy_text_arr, j, pre=False, n=3)

                for k in range(len(text_arr)):
                    pre_values_k = self._get_n_grams(text_arr, k, pre=True, n=3)
                    post_values_k = self._get_n_grams(text_arr, k, pre=False, n=3)
                    if self._is_ngram_similar(pre_values_j, pre_values_k) and self._is_ngram_similar(post_values_j, post_values_k):
                        noisy_value = noisy_text_arr[j]
                        actual_value = text_arr[k]
                        pre_values = self._get_n_grams(text_arr, k, pre=True, n=self.n)
                        post_values = self._get_n_grams(text_arr, k, pre=False, n=self.n)
                        if self.output_type == "word":
                            ngrams.append([pre_values, post_values, noisy_value, actual_value])
                        elif self.output_type == "character":
                            pre_values = [str(a) for a in pre_values]
                            post_values = [str(a) for a in post_values]

                            noisy_full_string = " ".join(pre_values) + " " + noisy_value + " " + " ".join(post_values)
                            actual_full_string = " ".join(pre_values) + " " + actual_value + " " + " ".join(post_values)
                            noisy_index = len(" ".join(pre_values)) + 1
                            for c in range(len(noisy_value)):
                                idx = c + noisy_index
                                new_pre_values = actual_full_string [idx-self.n:idx]
                                new_post_values = actual_full_string [idx+1:idx+self.n + 1]
                                new_noisy_values = noisy_full_string[idx]
                                new_actual_values = actual_full_string[idx]
                                ngrams.append([new_pre_values, new_post_values, new_noisy_values, new_actual_values])
        return ngrams
                
    def __getitem__(self, idx):
        pre_values, post_values, noisy_value, actual_value = self._data[0][idx]
        return pre_values, post_values, noisy_value, actual_value

    def __len__(self):
        return len(self._data[0])
