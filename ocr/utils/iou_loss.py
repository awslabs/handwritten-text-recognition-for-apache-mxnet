# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from mxnet.gluon.loss import Loss
from mxnet import gluon

class IOU_loss(Loss):
    r"""Calculates the iou between `pred` and `label`.

    Implementation based on:
        Yu, J., Jiang, Y., Wang, Z., Cao, Z., & Huang, T. (2016, October). Unitbox: An advanced object detection network.
            # In Proceedings of the 2016 ACM on Multimedia Conference (pp. 516-520). ACM.
            
    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.

    Inputs:
        - **pred**: prediction tensor with shape [x, y, w, h] each in percentages
        - **label**: target tensor with the shape [x, y, w, h] each in percentages
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: IOU loss tensor with shape (batch_size,).
    """

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(IOU_loss, self).__init__(weight, batch_axis, **kwargs)
        
    def hybrid_forward(self, F, pred, label, sample_weight=None):
        '''
        Interpreted from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        Steps to calculate IOU
        1) Calculate the area of the predicted and actual bounding boxes
        2) Calculate the area of the intersection between the predicting and actual bounding box
        3) Calculate the log IOU by: log(intersection area / (union area))
        3) If the bounding boxes do not overlap with one another, set the iou to zero
        4) Calculate the negative mean of the IOU
        '''
        
        pred_area = pred[:, 2] * pred[:, 3]
        label_area = label[:, 2] * label[:, 3]

        x1_1, y1_1 = pred[:, 0], pred[:, 1]
        x2_1, y2_1 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]

        x1_2, y1_2 = label[:, 0], label[:, 1]
        x2_2, y2_2 = label[:, 0] + label[:, 2], label[:, 1] + label[:, 3]

        x_overlaps = F.logical_or(x2_1 < x1_2, x1_1 > x2_2)
        y_overlaps = F.logical_or(y2_1 < y1_2, y1_1 > y2_2)
        overlaps = F.logical_not(F.logical_or(x_overlaps, y_overlaps))

        x1_1 = x1_1.expand_dims(0)
        y1_1 = y1_1.expand_dims(0)
        x2_1 = x2_1.expand_dims(0)
        y2_1 = y2_1.expand_dims(0)
        x1_2 = x1_2.expand_dims(0)
        y1_2 = y1_2.expand_dims(0)
        x2_2 = x2_2.expand_dims(0)
        y2_2 = y2_2.expand_dims(0)
        
        x_a = F.max(F.concat(x1_1, x1_2, dim=0), axis=0)
        y_a = F.max(F.concat(y1_1, y1_2, dim=0), axis=0)
        x_b = F.min(F.concat(x2_1, x2_2, dim=0), axis=0)
        y_b = F.min(F.concat(y2_1, y2_2, dim=0), axis=0)
        
        inter_area = (x_b - x_a) * (y_b - y_a)

        iou = F.log(inter_area) - F.log(pred_area + label_area - inter_area)
        
        loss = gluon.loss._apply_weighting(F, iou, self._weight, sample_weight)
        loss = F.where(F.logical_not(overlaps), F.zeros(shape=overlaps.shape), loss)
        mean_loss = F.mean(loss, axis=self._batch_axis, exclude=True)
        return -mean_loss
