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
    
def draw_box(bounding_box, image, line_type):
    image_h, image_w = image.shape[-2:]
    (x, y, w, h) = bounding_box
    (x1, y1, x2, y2) = (x, y, x + w, y + h)
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    if y2 >= image_h:
        y2 = image_h - 1
    if x2 >= image_w:
        x2 = image_w - 1
    if y1 >= image_h:
        y1 = image_h - 1
    if x1 >= image_w:
        x1 = image_w - 1

    image = draw_line(image, y1, x1, y2, x1, line_type)
    image = draw_line(image, y2, x1, y2, x2, line_type)
    image = draw_line(image, y2, x2, y1, x2, line_type)
    image = draw_line(image, y1, x2, y1, x1, line_type)
    return image

def draw_box_on_image(pred, label, data):    
    image_h, image_w = data.shape[-2:]
    pred[:, 0], pred[:, 1] = pred[:, 0] * image_w, pred[:, 1] * image_h
    pred[:, 2], pred[:, 3] = pred[:, 2] * image_w, pred[:, 3] * image_h

    label[:, 0], label[:, 1] = label[:, 0] * image_w, label[:, 1] * image_h
    label[:, 2], label[:, 3] = label[:, 2] * image_w, label[:, 3] * image_h

    for i in range(data.shape[0]):
        image = data[i, 0]
        image = draw_box(pred[i, :], image, line_type="full")
        image = draw_box(label[i, :], image, line_type="dotted")
        data[i, 0, :, :] = image
    return data
