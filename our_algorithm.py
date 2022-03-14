from filtering import bilateral_filter, canny_edge_detection, layer_separation
from color_transfer import convert_color_space_RGB_to_GRAY
import timeit
import numpy as np
import math


def increase_dim(e, I):
    if len(I.shape) > 2:
        if I.shape[2] == 3:
            t = np.stack((e, e, e), axis=2)
        elif I.shape[2] == 4:
            t = np.stack((e, e, e, e), axis=2)
    return t


def landscape_pictures(I):
    start = timeit.default_timer()
    m = round(math.sqrt(I.shape[0] * I.shape[1]) / 500)
    k = m * 3
    if k < 1:
        k = 1

    I = bilateral_filter(I, 2, 2 / 70, 2)

    I = layer_separation(I, 5)

    print('k =', k)
    for _ in range(3):
        I = bilateral_filter(I, k, k / 70, k)

    ip_gray = convert_color_space_RGB_to_GRAY(I)

    h = m * 3
    if h < 3:
        h = 3

    print('h =', h)

    c = canny_edge_detection(ip_gray, h)

    I = I - increase_dim(c, I)
    I = I.clip(0.0, 255.0).astype(np.uint8)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return I


def portrait_pictures(I):
    start = timeit.default_timer()
    m = round(math.sqrt(I.shape[0] * I.shape[1]) / 500)
    k = m * 5
    if k < 1:
        k = 1
    I = bilateral_filter(I, k, k / 100, k)

    I = layer_separation(I, 2)

    print('k =', k)
    for _ in range(3):
        I = bilateral_filter(I, k, k / 100, k)

    ip_gray = convert_color_space_RGB_to_GRAY(I)
    c = canny_edge_detection(ip_gray)

    I = I - increase_dim(c, I) * 0.1
    I = I.clip(0.0, 255.0)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return I

# if __name__ == '__main__':
#     img_path = './img_test/test1.jpg'

#     I = cv2.imread(img_path, cv2.COLOR_BGR2RGB).astype('float32')

#     I = landscape_pictures(I)
#     # I = portrait_pictures(I)
#     cv2.imwrite('./img_test/_result.jpg', I)
