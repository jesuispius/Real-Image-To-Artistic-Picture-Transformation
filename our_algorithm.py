# ==================================================================================================================== #
# Collaborators: Phuoc Nguyen + Tri Le
# Description: This file includes the algorithms to be blending and flattening the image by our own.
# Filename: our_algorithm.py
# ==================================================================================================================== #


# Dependencies
from filtering import bilateral_filter, canny_edge_detection
from color_transfer import convert_color_space_RGB_to_GRAY
import timeit
import numpy as np
import math


def increase_dim(e, I):
    '''
    increase the dims of e to agree with I
    '''
    if len(I.shape) > 2:

        if I.shape[2] == 3:
            t = np.stack((e, e, e), axis=2)
        elif I.shape[2] == 4:
            t = np.stack((e, e, e, e), axis=2)

    return t


def landscape(I):
    '''
    Generate artistic image for lanscape picture
    '''
    start = timeit.default_timer()

    k = 5
    for _ in range(2):
        I = bilateral_filter(I, k, k / 50, k)

    ip_gray = convert_color_space_RGB_to_GRAY(I)

    c = canny_edge_detection(ip_gray)

    I = I - increase_dim(c, I)
    I = I.clip(0.0, 255.0).astype(np.uint8)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return I


def portrait(I):
    '''
    Generate artistic image for portait picture
    '''
    start = timeit.default_timer()
  
    k = 7
    for _ in range(3):
        I = bilateral_filter(I, k, k / 75, k) 
        k = k - 2

    ip_gray = convert_color_space_RGB_to_GRAY(I)
    c = canny_edge_detection(ip_gray)
  
    I = I - increase_dim(c, I) * 0.1
    I = I.clip(0.0, 255.0)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return I


# if __name__ == '__main__':
#     import cv2
#     img_path = './img_test/test0.jpg'

#     I = cv2.imread(img_path, cv2.COLOR_BGR2RGB).astype('float32')

#     for k in range(1, 6):

#         for j in range(20, 121,20):
#             I2 = bilateral_filter(I.copy(), k, 1 / j, 5)
#             cv2.imwrite(
#                 './img_test/test/result{0}_{1}.jpg'.format(k, j, I2), I2)
