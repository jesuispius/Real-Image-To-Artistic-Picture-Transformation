import numpy as np
import math


def MedianFiltering(img, kernel_size):
    output_img = np.zeros_like(img)
    edge = math.floor(kernel_size / 2)

    for x in range(edge, img.shape[0] - edge):
        for y in range(edge, img.shape[1] - edge):
            mask_img = img[x - edge: x + edge + 1, y - edge: y + edge + 1]
            median = np.sort(np.ravel(mask_img))
            output_img[x, y] = median[int(kernel_size * kernel_size / 2)]

    return output_img


