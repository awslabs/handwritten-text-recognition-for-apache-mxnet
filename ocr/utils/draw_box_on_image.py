# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from skimage.draw import line_aa
import matplotlib.pyplot as plt

def draw_line(image, y1, x1, y2, x2, line_type):
    rr, cc, val = line_aa(y1, x1, y2, x2)
    if line_type == "dotted":
        rr = np.delete(rr, np.arange(0, rr.size, 5))
        cc = np.delete(cc, np.arange(0, cc.size, 5))
    image[rr, cc] = 0
    return image
    
def draw_box(bounding_box, image, line_type, is_xywh=True):
    image_h, image_w = image.shape[-2:]
    if is_xywh:
        (x, y, w, h) = bounding_box
        (x1, y1, x2, y2) = (x, y, x + w, y + h)
    else:
        (x1, y1, x2, y2) = bounding_box
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    if y2 >= image_h:
        y2 = image_h - 1
    if x2 >= image_w:
        x2 = image_w - 1
    if y1 >= image_h:
        y1 = image_h - 1
    if x1 >= image_w:
        x1 = image_w - 1
    if y2 < 0:
        y2 = 0
    if x2 < 0:
        x2 =0
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0

    image = draw_line(image, y1, x1, y2, x1, line_type)
    image = draw_line(image, y2, x1, y2, x2, line_type)
    image = draw_line(image, y2, x2, y1, x2, line_type)
    image = draw_line(image, y1, x2, y1, x1, line_type)
    return image

def draw_boxes_on_image(pred, label, images):
    ''' Function to draw multiple bounding boxes on the images. Predicted bounding boxes will be
    presented with a dotted line and actual boxes are presented with a solid line.

    Parameters
    ----------
    
    pred: [n x [x, y, w, h]]
        The predicted bounding boxes in percentages. 
        n is the number of bounding boxes predicted on an image

    label: [n x [x, y, w, h]]
        The actual bounding boxes in percentages
        n is the number of bounding boxes predicted on an image

    images: [[np.array]]
        The correponding images.

    Returns
    -------

    images: [[np.array]]
        Images with bounding boxes printed on them.
    '''
    image_h, image_w = images.shape[-2:]
    label[:, :, 0], label[:, :, 1] = label[:, :, 0] * image_w, label[:, :, 1] * image_h
    label[:, :, 2], label[:, :, 3] = label[:, :, 2] * image_w, label[:, :, 3] * image_h
    for i in range(len(pred)):
        pred_b = pred[i]
        pred_b[:, 0], pred_b[:, 1] = pred_b[:, 0] * image_w, pred_b[:, 1] * image_h
        pred_b[:, 2], pred_b[:, 3] = pred_b[:, 2] * image_w, pred_b[:, 3] * image_h

        image = images[i, 0]
        for j in range(pred_b.shape[0]):
            image = draw_box(pred_b[j, :], image, line_type="dotted")

        for k in range(label.shape[1]):
            image = draw_box(label[i, k, :], image, line_type="solid")
        images[i, 0, :, :] = image
    return images

def draw_box_on_image(pred, label, images):
    ''' Function to draw bounding boxes on the images. Predicted bounding boxes will be
    presented with a dotted line and actual boxes are presented with a solid line.

    Parameters
    ----------
    
    pred: [[x, y, w, h]]
        The predicted bounding boxes in percentages

    label: [[x, y, w, h]]
        The actual bounding boxes in percentages

    images: [[np.array]]
        The correponding images.

    Returns
    -------

    images: [[np.array]]
        Images with bounding boxes printed on them.
    '''

    image_h, image_w = images.shape[-2:]
    pred[:, 0], pred[:, 1] = pred[:, 0] * image_w, pred[:, 1] * image_h
    pred[:, 2], pred[:, 3] = pred[:, 2] * image_w, pred[:, 3] * image_h

    label[:, 0], label[:, 1] = label[:, 0] * image_w, label[:, 1] * image_h
    label[:, 2], label[:, 3] = label[:, 2] * image_w, label[:, 3] * image_h

    for i in range(images.shape[0]):
        image = images[i, 0]
        image = draw_box(pred[i, :], image, line_type="dotted")
        image = draw_box(label[i, :], image, line_type="solid")
        images[i, 0, :, :] = image
    return images
