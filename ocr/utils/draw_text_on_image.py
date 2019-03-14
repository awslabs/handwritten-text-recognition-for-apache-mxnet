# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

def draw_text_on_image(images, text):
    output_image_shape = (images.shape[0], images.shape[1], images.shape[2] * 2, images.shape[3])  # Double the output_image_shape to print the text in the bottom
    
    output_images = np.zeros(shape=output_image_shape)
    for i in range(images.shape[0]):
        white_image_shape = (images.shape[2], images.shape[3])
        white_image = np.ones(shape=white_image_shape)*1.0
        text_image = cv2.putText(white_image, text[i], org=(5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0.0, thickness=1)
        output_images[i, :, :images.shape[2], :] = images[i]
        output_images[i, :, images.shape[2]:, :] = text_image
    return output_images
