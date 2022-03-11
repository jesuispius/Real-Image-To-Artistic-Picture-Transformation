from upload_image import upload_image, read_image
from filtering import bilateral_filter, median_filter, canny_edge_detection, sobel_filter
from color_transfer import convert_color_space_RGB_to_GRAY, convert_color_space_RGB_to_BGR
import timeit
import cv2
import numpy as np


def increase_dim(e, I):
    if len(I.shape) > 2:
        if I.shape[2] == 3:
            t = np.stack((e, e, e), axis=2)
        elif I.shape[2] == 4:
            t = np.stack((e, e, e, e), axis=2)
    return t


def landscape_pictures(I):
    start = timeit.default_timer()
    k = 7
    for i in range(3):
        I = bilateral_filter(I, k, k / 100, k)

    ip_gray = convert_color_space_RGB_to_GRAY(I)

    c = canny_edge_detection(ip_gray)
    I = I - increase_dim(c, I)
    I = I.clip(0.0, 255.0).astype(np.uint8)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return I


def portrait_pictures(I):
    start = timeit.default_timer()
    k = 5
    for i in range(5):
        I = bilateral_filter(I, k, k/100, k)

    ip_gray = convert_color_space_RGB_to_GRAY(I)
    
    c = canny_edge_detection(ip_gray)
    I = I - increase_dim(c,I)*0.1
    I = I.clip(0.0, 255.0).astype(np.uint8)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return I

if __name__ == '__main__':

    I = cv2.imread('./img_test/test3.jpg',
                cv2.COLOR_BGR2RGB).astype('float32')  

    
    I = portrait_pictures(I)
    

    cv2.imwrite('./img_test/test3_result.jpg', I)


    # I = cv2.imread('./img_test/psu1.jpg',
    #             cv2.COLOR_BGR2RGB).astype('float32')  

    
    # I = landscape_pictures(I)
    

    # cv2.imwrite('./img_test/psu1_result.jpg', I)   
