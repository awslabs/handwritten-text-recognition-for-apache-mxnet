import cv2
import numpy as np

def draw_text_on_image(images, text):
    output_images = np.zeros(shape=images.shape)
    for i in range(images.shape[0]):
        image = cv2.putText(images[i, 0, :], text[i], org=(5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0, thickness=3)
        output_images[i, 0, :] = image
    return output_images
