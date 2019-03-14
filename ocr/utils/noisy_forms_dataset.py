# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pickle
import os

from .iam_dataset import IAMDataset

class Noisy_forms_dataset:
    '''
    The noisy_forms_dataset provides pairs of identical passages, one of the passages is noisy.
    The noise includes random replacements, insertions and deletions.
    
    Parameters
    ----------
    noise_source_transform: (np.array, str) -> str
        The noise_source_transform is a function that takes an image containing a single line of handwritten text
        and outputs a noisy version of the text string.

    train: bool
        Indicates if the data should be used for training or testing

    name: str
        An identifier to save a temporary version of the database
    '''
    train_size = 0.8
    def __init__(self, noise_source_transform, train, name, topK_decode):
        self.noise_source_transform = noise_source_transform
        self.iam_dataset_form = IAMDataset("form", output_data="text", train=True)
        self.iam_dataset_line = IAMDataset("line", output_data="text", train=True)
        self.train = train
        self.seed = np.random.uniform(0, 1.0, size=len(self.iam_dataset_form))
        self.topK_decode = topK_decode
        root = os.path.join("dataset", "noisy_forms")
        if not os.path.isdir(root):
            os.makedirs(root)
    
        ds_location = os.path.join(root, "ns{}.pickle".format(name))
        if os.path.isfile(ds_location):
            self.train_data, self.test_data = pickle.load(open(ds_location, 'rb'))
        else:
            self.train_data, self.test_data = self._get_data()
            pickle.dump((self.train_data, self.test_data), open(ds_location, 'wb'), protocol=2)

    def _is_line_in_form(self, line_text, form_text, p=0.8):
        '''
        Helper function to check if a line of text is within a form.
        Since there are differences in punctuations, spaces, etc. The line was split into separate words and if
        more than probability of the line is within the form, it's considered in the form.
        
        Parameters
        ----------
        line_text: str
            A string of a line.
        
        form_text: str
            A string of a whole passage.

        p: float, default=0.8
            the probability of words of a line that is within a form to consider the line is in the form.

        Return
        ------
        is_line_in_form: bool
            If the line is considered in the form, return true, return false otherwise.
        '''
        line_text_array = np.array(line_text.split(" "))
        form_text_array = np.array(form_text.split(" "))
        in_form = np.in1d(line_text_array, form_text_array)
        if np.mean(in_form) > p:
            return True
        else:
            return False
            
    def _get_data(self):
        '''
        Generates a noisy text using the noise_source_transform then organises the data from multiple lines 
        into a single form (to keep the context of the form consistent).
        
        Returns
        -------
        train_data: [(str, str)]
            Contains a list of tuples that contains a two passages that are the same but one is noisy.

        test_data: [(str, str)]
            Contains a list of tuples that contains a two passages that are the same but one is noisy. This
            list of tuples contains independent samples compared to train_data.
        '''
        train_data = []
        test_data = []
                
        for idx_form in range(len(self.iam_dataset_form)):
            print("{}/{}".format(idx_form, len(self.iam_dataset_form)))
            _, form_text = self.iam_dataset_form[idx_form]
            form_text = form_text[0].replace("\n", " ")

            _, full_form_text = self.iam_dataset_form[idx_form]
            full_form_text = full_form_text[0].replace("\n", " ")

            lines_in_form = []
            for idx_line in range(len(self.iam_dataset_line)):
                # Iterates through every line data to check if it's within the form.
                image, line_text = self.iam_dataset_line[idx_line]
                line_text = line_text[0]
                
                if self._is_line_in_form(line_text, form_text):
                    prob = self.noise_source_transform(image, line_text)
                    predicted_text = self.topK_decode(np.argmax(prob, axis=2))[0]
                    lines_in_form.append(predicted_text)
                    form_text = form_text.replace(line_text, "")

            predicted_form_text = ' '.join(lines_in_form)
            if len(predicted_text) > 500:
                import pdb; pdb.set_trace();                        

            if self.seed[idx_form] < self.train_size:
                train_data.append([predicted_form_text, full_form_text])
            else:
                test_data.append([predicted_form_text, full_form_text])

        return train_data, test_data
        
    def __getitem__(self, idx):
        if self.train:
            noisy_text, actual_text = self.train_data[idx]
            return noisy_text, actual_text
        else:
            noisy_text, actual_text = self.test_data[idx]
            return noisy_text, actual_text

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
