import numpy as np
import os
import pickle
from mxnet.gluon.data import dataset

class Ngram_dataset(dataset.ArrayDataset):
    '''
    The ngram_dataset takes a noisy_forms_dataset object and converts it into an ngram dataset.
    That is, for each word i a tuple is created that contains:
       * n words before
       * n words after
       * Noisy word i
       * The actual word i

    Parameters
    ----------
    passage_ds: Noisy_forms_dataset
        A noisy_forms_dataset object

    name: 

    '''
    def __init__(self, passage_ds, name, n=3):
        self.passage_ds = passage_ds
        self.n = n
        root = os.path.join("dataset", "noisy_forms")
        if not os.path.isdir(root):
            os.makedirs(root)
        ds_location = os.path.join(root, "ngram_{}.pickle".format(name))
        # if os.path.isfile(ds_location):
        #     self.ngram_data = pickle.load(open(ds_location, 'rb'))
        # else:
        #     self.ngram_data = self._get_data()
        #     pickle.dump(self.ngram_data, open(ds_location, 'wb'), protocol=2)
        data = self._get_data()
        super(Ngram_dataset, self).__init__(data)

    def _get_n_grams(self, text_arr, idx, pre):
        output = []
        if pre:
            indexes = range(idx - self.n, idx)
        else:
            indexes = range(idx + 1, idx + self.n + 1)
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
                pre_values_j = self._get_n_grams(text_arr, j, pre=True)
                post_values_j = self._get_n_grams(text_arr, j, pre=False)

                for k in range(len(text_arr)):
                    pre_values_k = self._get_n_grams(text_arr, k, pre=True)
                    post_values_k = self._get_n_grams(text_arr, k, pre=False)

                    if self._is_ngram_similar(pre_values_j, pre_values_k) and self._is_ngram_similar(post_values_j, post_values_k):
                        noisy_value = noisy_text_arr[j]
                        actual_value = text_arr[k]
                        ngrams.append([pre_values_j, post_values_j, noisy_value, actual_value])
        return ngrams
                
    def __getitem__(self, idx):
        pre_values, post_values, noisy_value, actual_value = self._data[0][idx]
        return pre_values, post_values, noisy_value, actual_value

    # def __len__(self):
    #     return len(self._data)
