# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from os import path
import zipfile 

import mxnet as mx

dirname = 'dataset'
if not path.isdir(dirname):
    os.makedirs(dirname)
    
dirname = 'models'
if not path.isdir(dirname):
    os.makedirs(dirname)
    
print("Downloading Paragraph Segmentation parameters")
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/paragraph_segmentation2.params', dirname=dirname)

print("Downloading Word Segmentation parameters")
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/word_segmentation2.params', dirname=dirname)

print("Downloading Handwriting Line Recognition parameters")
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/handwriting_line8.params', dirname=dirname)

print("Downloading Denoiser parameters")
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/denoiser2.params', dirname=dirname)

print("Downloading cost matrices")
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/deletion_costs.txt', dirname=dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_costs.txt', dirname=dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/insertion_costs.txt', dirname=dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/substitute_probs.json', dirname=dirname)

print("Downloading fonts")
dirname = path.join('dataset','fonts')
if not path.isdir(dirname):
    os.makedirs(dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/fonts.zip', dirname=dirname)
with zipfile.ZipFile(path.join(dirname, "fonts.zip"),"r") as zip_ref:
    zip_ref.extractall(dirname)

print("Downloading text datasets")
dirname = path.join('dataset','typo')
if not path.isdir(dirname):
    os.makedirs(dirname)
    
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/alicewonder.txt', dirname=dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/all.txt', dirname=dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/text_train.txt', dirname=dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/validating.json', dirname=dirname)
mx.test_utils.download('https://s3.us-east-2.amazonaws.com/gluon-ocr/models/typo-corpus-r1.txt', dirname=dirname)
