# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

def expand_bounding_box(bb, expand_bb_scale_x=0.05, expand_bb_scale_y=0.05):
    (x, y, w, h) = bb
    new_w = (1 + expand_bb_scale_x) * w
    new_h = (1 + expand_bb_scale_y) * h
        
    x = x - (new_w - w)/2
    y = y - (new_h - h)/2
    w = new_w
    h = new_h
    return (x, y, w, h)
