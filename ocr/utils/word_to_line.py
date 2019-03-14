# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from scipy.cluster.hierarchy import fcluster

from .expand_bounding_box import expand_bounding_box

def _clip_value(value, max_value):
    '''
    Helper function to make sure that "value" will not be greater than max_value
    or lower than 0.
    '''
    output = value
    if output < 0:
        output = 0
    if output > max_value:
        output = max_value
    return int(output)

def _get_max_coord(bbs, x_or_y):
    '''
    Helper function to find the largest coordinate given a list of
    bounding boxes in the x or y direction.
    '''
    assert x_or_y in ["x", "y"], "x_or_y can only be x or y"
    max_value = 0.0
    for bb in bbs:
        if x_or_y == "x":
            value = bb[0] + bb[2]
        else:
            value = bb[1] + bb[3]
        if value > max_value:
            max_value = value
    return max_value

def _get_min_coord(bbs, x_or_y):
    '''
    Helper function to find the largest coordinate given a list of
    bounding boxes in the x or y direction.
    '''
    assert x_or_y in ["x", "y"], "x_or_y can only be x or y"
    min_value = 100
    for bb in bbs:
        if x_or_y == "x":
            value = bb[0]
        else:
            value = bb[1]
        if value < min_value:
            min_value = value
    return min_value

def _get_bounding_box_of_bb_list(bbs_in_a_line):
    '''
    Given a list of bounding boxes, find the maximum x, y and
    minimum x, y coordinates. This is the bounding box that
    emcompasses all the words. Return this bounding box in the form
    (x', y', w', h').
    '''    
    max_x = _get_max_coord(bbs_in_a_line, x_or_y="x")
    min_x = _get_min_coord(bbs_in_a_line, x_or_y="x")

    max_y = _get_max_coord(bbs_in_a_line, x_or_y="y")
    min_y = _get_min_coord(bbs_in_a_line, x_or_y="y")
            
    line_bb = (min_x, min_y, max_x - min_x, max_y - min_y)
    return line_bb

def _filter_bbs(bbs, min_size=0.005):
    '''
    Remove bounding boxes that are too small 
    '''
    output_bbs = []
    for bb in bbs:
        if bb[2] * bb[3] > min_size:
            output_bbs.append(bb)
    return np.array(output_bbs)

def _get_line_overlap_percentage(y1, h1, y2, h2):
    '''
    Calculates how much (percentage) y2->y2+h2 overlaps with y1->y1+h1.
    Algorithm assumes that y2 is larger than y1
    '''
    if y2 > y1 and (y1 + h1) > y2:
        # Is y2 enclosed in y1
        if (y1 + h1) > (y2 + h2):
            return 1.0
        else:
            return ((y1 + h1) - (y2))/h1
    else:
        return 0.0
    
def _get_rect_overlap_percentage(x1, y1, w1, h1, x2, y2, w2, h2):
    '''
    Calculate how much (in percentage) that rect2 overlaps with rect1
    '''
    # Check if rect overlaps
    x_overlap = (x1 + w1 >= x2 and x2 >= x1) or (x2 + w2 >= x1 and x1 >= x2)
    y_overlap = (y1 + h1 >= y2 and y2 >= y1) or (y2 + h2 >= y1 and y1 >= y2)
    if x_overlap and y_overlap:
        intersect_size = max(0, min(x1 + w1, x2 + w2) - min(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        s1 = w1 * h1
        return intersect_size / s1
    else:
        return 0
    
def combine_bbs_into_lines(bbs, y_overlap=0.2):
    '''
    Algorithm to group word crops into lines.
    Iterates over every bb, if the overlap in the y direction
    between 2 boxes has less than y_overlap overlap, then group the previous words together.
    '''
    line_bbs = []
    bbs_in_a_line = []
    y_indexes = np.argsort(bbs[:, 1])
    # Iterate through the sorted bounding box.
    previous_y_coords = None
    for y_index in y_indexes:
        y_coords = (bbs[y_index, 1], bbs[y_index, 3]) # y and height
        
        # new line if the overlap is more than y_overlap
        if previous_y_coords is not None:
            line_overlap_percentage1 = _get_line_overlap_percentage(
                previous_y_coords[0], previous_y_coords[1],
                y_coords[0], y_coords[1])
            line_overlap_percentage2 = _get_line_overlap_percentage(
                y_coords[0], y_coords[1],
                previous_y_coords[0], previous_y_coords[1])
            line_overlap_percentage = max(line_overlap_percentage1, line_overlap_percentage2)
            if line_overlap_percentage < y_overlap:   
                line_bb = _get_bounding_box_of_bb_list(bbs_in_a_line)
                line_bbs.append(line_bb)
                bbs_in_a_line = []
        bbs_in_a_line.append(bbs[y_index, :])
        previous_y_coords = y_coords
    
    # process the last line
    line_bb = _get_bounding_box_of_bb_list(bbs_in_a_line)
    line_bbs.append(line_bb)
    return line_bbs
        
def sort_bbs_line_by_line(bbs, y_overlap=0.2):
    '''
    Function to combine word bbs into lines.
    '''
    line_bbs = _filter_bbs(bbs, min_size=0.0001) #Filter small word BBs
    line_bbs = combine_bbs_into_lines(line_bbs, y_overlap)
    line_bb_expanded = []
    for line_bb in line_bbs:
        line_bb_i = expand_bounding_box(line_bb, expand_bb_scale_x=0.1,
                                              expand_bb_scale_y=0.05)
        line_bb_expanded.append(line_bb_i)
    line_bbs = np.array(line_bb_expanded)
    
    # X start heuristics
    # Remove lines that start more than 150% away
    x_start_within_boundary = line_bbs[:, 0] < 0.5
    line_bbs = line_bbs[x_start_within_boundary]
    
    # Remove lines that start 20% away from the average
    x_start_line_bbs = line_bbs[:, 0]
    x_start_diff = np.abs(x_start_line_bbs - np.median(x_start_line_bbs))
    x_start_remove = x_start_diff < 0.2
    line_bbs = line_bbs[x_start_remove]
    
    # X length heuristics
    # Remove lines that are 50% shorter excluding the last element
    if len(line_bbs) > 1:
        x_length_line_bbs = line_bbs[:-1, 0] - line_bbs[:-1, 2]
        x_length_diff = np.abs(x_length_line_bbs - np.median(x_length_line_bbs))    
        x_length_remove = x_length_diff < 0.5
        last_line = line_bbs[-1]
        line_bbs = line_bbs[:-1][x_length_remove]
        line_bbs = np.vstack([line_bbs, last_line])
    
    # Y height heuristics
    # Split lines that are more than 1.5 of the others
    y_height = line_bbs[:, 3]
    y_height_diff = np.abs(y_height/np.median(y_height))
    y_height_remove = y_height_diff > 1.65
    
    new_line_bbs = []
    for i in range(line_bbs.shape[0]):
        if y_height_remove[i]:
            # split line into 2
            new_line_top = np.copy(line_bbs[i])
            new_line_top[3] = new_line_top[3] / 2
            
            new_line_bottom = np.copy(line_bbs[i])
            new_line_bottom[1] = new_line_bottom[1] + new_line_bottom[3]/2
            new_line_bottom[3] = new_line_bottom[3] / 2
                        
            new_line_bbs.append(new_line_top)
            new_line_bbs.append(new_line_bottom)
        else:
            new_line_bbs.append(line_bbs[i])
    line_bbs = np.vstack(new_line_bbs)
    
    # Y consistency heuristics
    # Remove lines that overlap by 40% with other lines
    line_total_overlap = []
    for i in range(line_bbs.shape[0]):
        overlap_i = 0.0
        for j in range(line_bbs.shape[0]):
            if i != j:
                line_i, line_j = line_bbs[i], line_bbs[j]
                overlap_i += _get_rect_overlap_percentage(line_i[0], line_i[1], line_i[2], line_i[3],
                                                          line_j[0], line_j[1], line_j[2], line_j[3])
        line_total_overlap.append(overlap_i)
    overlap_remove = np.array(line_total_overlap) < 1
    line_bbs = line_bbs[overlap_remove]
    return line_bbs

def crop_line_images(image, line_bbs):
    '''
    Given the input form image, crop the image given a list of bounding boxes.
    '''
    line_images = []
    for line_bb in line_bbs:
        (x, y, w, h) = line_bb
        image_h, image_w = image.shape[-2:]
        (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
        x1 = _clip_value(x, max_value=image_w)
        x2 = _clip_value(x + w, max_value=image_w)
        y1 = _clip_value(y, max_value=image_h)
        y2 = _clip_value(y + h, max_value=image_h)
        
        line_image = image[y1:y2, x1:x2]    
        if line_image.shape[0] > 0 and line_image.shape[1] > 0:
            line_images.append(line_image)
    return line_images
