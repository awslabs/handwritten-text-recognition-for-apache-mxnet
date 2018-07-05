#!/bin/bash

python paragraph_segmentation_dcnn.py -r 0.001 -e 301 -n cnn_mse.params
python paragraph_segmentation_dcnn.py -r 0.0001 -l iou -e 150 -n cnn_iou.params -f cnn_mse.params
