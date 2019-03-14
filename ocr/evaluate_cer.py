# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import difflib
import logging
import math
import string
import random

import numpy as np
import mxnet as mx
from tqdm import tqdm


from .paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from .word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from .handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from .handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

from .utils.iam_dataset import IAMDataset, crop_handwriting_page
from .utils.sclite_helper import ScliteHelper
from .utils.word_to_line import sort_bbs_line_by_line, crop_line_images

# Setup
logging.basicConfig(level=logging.DEBUG)
random.seed(123)
np.random.seed(123)
mx.random.seed(123)

# Input sizes
form_size = (1120, 800)
segmented_paragraph_size = (800, 800)
line_image_size = (60, 800)

# Parameters
min_c = 0.01
overlap_thres = 0.001
topk = 400
rnn_hidden_states = 512
rnn_layers = 2
max_seq_len = 160

recognition_model = "models/handwriting_line8.params.params"
paragraph_segmentation_model = "models/paragraph_segmentation2.params"
word_segmentation_model = "models/word_segmentation2.params"

def get_arg_max(prob):
    '''
    The greedy algorithm convert the output of the handwriting recognition network
    into strings.
    '''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]

denoise_func = get_arg_max

if __name__ == '__main__':
    
    # Compute context
    ctx = mx.gpu(1)

    # Models
    logging.info("Loading models...")
    paragraph_segmentation_net = ParagraphSegmentationNet(ctx)
    paragraph_segmentation_net.load_parameters(paragraph_segmentation_model, ctx)

    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters(word_segmentation_model, ctx)

    handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=rnn_hidden_states,
                                                                 rnn_layers=rnn_layers,
                                                                 max_seq_len=max_seq_len,
                                                                 ctx=ctx)
    handwriting_line_recognition_net.load_parameters(recognition_model, ctx)
    logging.info("models loaded.")

    # Data
    logging.info("loading data...")
    test_ds = IAMDataset("form_original", train=False)
    logging.info("data loaded.")


    sclite = ScliteHelper()
    for i in tqdm(range(len(test_ds))):
        image, text = test_ds[i]
        resized_image = paragraph_segmentation_transform(image, image_size=form_size)
        paragraph_bb = paragraph_segmentation_net(resized_image.as_in_context(ctx))
        paragraph_segmented_image = crop_handwriting_page(image, paragraph_bb[0].asnumpy(), image_size=segmented_paragraph_size)
        word_bb = predict_bounding_boxes(word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)
        line_bbs = sort_bbs_line_by_line(word_bb)
        line_images = crop_line_images(paragraph_segmented_image, line_bbs)

        predicted_text = []
        for line_image in line_images:
            line_image = handwriting_recognition_transform(line_image, line_image_size)
            character_probabilities = handwriting_line_recognition_net(line_image.as_in_context(ctx))
            decoded_text = denoise_func(character_probabilities)
            predicted_text.append(decoded_text)

        actual_text = text[0].replace("&quot", '\"').replace("&amp", "&").replace('";', '\"')[:-1]
        actual_text = actual_text.split("\n")
        if len(predicted_text) > len(actual_text):
            predicted_text = predicted_text[:len(actual_text)]
        sclite.add_text(predicted_text, actual_text)

    _, er = sclite.get_cer()
    print("Mean CER = {}".format(er))
