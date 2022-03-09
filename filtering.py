# ==================================================================================================================== #
# Collaborators: Phuoc Nguyen + Tri Le
# Description: This file includes the function to upload the image from devices to webpage
# Filename: upload_image.py
# ==================================================================================================================== #

# Dependencies
import os
import numpy as np
import math
from multiprocessing import Pool
import pickle
from scipy import signal

# Global Variable
TWO_PI = 2.0 * math.pi
MAX_PROCESS_NUM = 2
EPSILON = 1e-8


# -------------------------------------------------------------------------------------------------------------------- #
# MEDIAN FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def MedianFiltering(img, kernel_size):
    """
    Median Filtering Function.

    :param img: image
    :param kernel_size: kernel size of filter
    :return: the image after applying median filter
    """
    # Initialisation
    output_img = np.zeros_like(img)
    edge = math.floor(kernel_size / 2)

    for x in range(edge, img.shape[0] - edge):
        for y in range(edge, img.shape[1] - edge):
            # Taking out the image matrix corresponding to the kernel size
            mask_img = img[x - edge: x + edge + 1, y - edge: y + edge + 1]

            # Sort the pixel in ascending order
            median = np.sort(np.ravel(mask_img))

            # Getting the median value from the sorted list
            median_val = int(kernel_size * kernel_size / 2)
            output_img[x, y] = median[median_val]

    return output_img


# -------------------------------------------------------------------------------------------------------------------- #
# BILATERAL FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def run_bilateral_filter(start_col, end_col, window_width, thread_id, input_image, sigma_space, sigma_intensity):
    def gaussian_kernel(r2, sigma):
        return (np.exp(-0.5 * r2 / sigma ** 2) * 3).astype(int) * 1.0 / 3.0

    sum_fr = np.ones(input_image.shape) * EPSILON
    sum_gs_fr = input_image * EPSILON

    for w_col in range(start_col, end_col):
        for w_row in range(-window_width, window_width + 1):
            gs = gaussian_kernel(w_col ** 2 + w_row ** 2, sigma_space)

            w_image = np.roll(input_image, [w_row, w_col], axis=[0, 1])

            fr = gs * gaussian_kernel((w_image - input_image) ** 2, sigma_intensity)

            sum_gs_fr += w_image * fr
            sum_fr += fr

    pickle.dump(sum_fr, open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bilateralsum_fr{0}.tmp'.format(thread_id)), 'wb'),
                pickle.HIGHEST_PROTOCOL)
    pickle.dump(sum_gs_fr, open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bilateralsum_gs_fr{0}.tmp'.format(thread_id)), 'wb'),
                pickle.HIGHEST_PROTOCOL)


def bilateral_filter(input_image, sigma_space=10.0, sigma_intensity=0.1, radius_window_width=1):
    responses = []

    pool = Pool(processes=MAX_PROCESS_NUM)

    windows_width = radius_window_width
    total_window_length = 2 * windows_width + 1

    rows_every_workers = total_window_length // MAX_PROCESS_NUM
    start_row = -windows_width
    end_row = start_row + rows_every_workers

    data = input_image.astype(np.float32) / 255.0

    for r in range(0, MAX_PROCESS_NUM):
        args = (start_row, end_row, windows_width, r,
                data, sigma_space, sigma_intensity)
        res = pool.apply_async(run_bilateral_filter, args)

        responses.append(res)

        start_row += rows_every_workers

        if r == MAX_PROCESS_NUM - 2:
            end_row = windows_width + 1
        else:
            end_row += rows_every_workers

    for res in responses:
        res.wait()

    sum_fr = None
    sum_gs_fr = None
    for thread_id in range(0, MAX_PROCESS_NUM):

        if thread_id == 0:
            sum_fr = pickle.load(
                open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'bilateralsum_fr{0}.tmp'.format(thread_id)), "rb"))
        else:
            sum_fr += pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'bilateralsum_fr{}.tmp'.format(thread_id)), "rb"))

        if thread_id == 0:
            sum_gs_fr = pickle.load(
                open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'bilateralsum_gs_fr{0}.tmp'.format(thread_id)), "rb"))
        else:
            sum_gs_fr += pickle.load(
                open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'bilateralsum_gs_fr{}.tmp'.format(thread_id)), "rb"))

        os.remove(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'bilateralsum_fr{0}.tmp'.format(thread_id)))
        os.remove(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'bilateralsum_gs_fr{0}.tmp'.format(thread_id)))

    sum_gs_fr = sum_gs_fr / sum_fr

    return (sum_gs_fr * 255.0).clip(0.0, 255.0).astype(np.uint8)


# -------------------------------------------------------------------------------------------------------------------- #
# MEAN FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def run_mean_filter(start_col, end_col, window_width, thread_id, input_image):
    sum_fr = np.zeros(input_image.shape)

    for w_col in range(start_col, end_col):
        for w_row in range(-window_width, window_width + 1):
            w_image = np.roll(input_image, [w_row, w_col], axis=[0, 1])
            sum_fr += w_image

    pickle.dump(sum_fr,
                open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mean_sum_fr{0}.tmp'.format(thread_id)),
                     'wb'),
                pickle.HIGHEST_PROTOCOL)


def mean_filter(input_image, radius_window_width=1):
    responses = []

    pool = Pool(processes=MAX_PROCESS_NUM)

    windows_width = radius_window_width
    total_window_length = 2 * windows_width + 1

    rows_every_workers = total_window_length // MAX_PROCESS_NUM
    start_row = -windows_width
    end_row = start_row + rows_every_workers

    data = input_image

    for r in range(0, MAX_PROCESS_NUM):
        args = (start_row, end_row, windows_width, r,
                data)
        res = pool.apply_async(run_mean_filter, args)

        responses.append(res)

        start_row += rows_every_workers

        if r == MAX_PROCESS_NUM - 2:
            end_row = windows_width + 1
        else:
            end_row += rows_every_workers

    for res in responses:
        res.wait()

    sum_fr = None

    for thread_id in range(0, MAX_PROCESS_NUM):

        if thread_id == 0:
            sum_fr = pickle.load(
                open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mean_sum_fr{0}.tmp'.format(thread_id)),
                     "rb"))
        else:
            sum_fr += pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'mean_sum_fr{}.tmp'.format(thread_id)), "rb"))

        os.remove(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'mean_sum_fr{0}.tmp'.format(thread_id)))

    sum_fr = sum_fr / (total_window_length ** 2)

    return sum_fr.clip(0.0, 255.0).astype(np.uint8)


# -------------------------------------------------------------------------------------------------------------------- #
# GAUSSIAN FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def run_gaussian_filter(start_col, end_col, window_width, thread_id, input_image, sigma_space):
    def gaussian_kernel(data, sigma):
        return (1 / (TWO_PI * sigma * sigma)) * np.exp(-((data) / (2.0 * sigma ** 2)))

    sum_fr = np.zeros(input_image.shape)

    for w_col in range(start_col, end_col):
        for w_row in range(-window_width, window_width + 1):
            gs = gaussian_kernel(w_col ** 2 + w_row ** 2, sigma_space)

            w_image = np.roll(input_image, [w_row, w_col], axis=[0, 1])

            sum_fr += gs * w_image

    pickle.dump(sum_fr, open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaussian_sum_fr{0}.tmp'.format(thread_id)), 'wb'),
                pickle.HIGHEST_PROTOCOL)


def gaussian_filter(input_image, sigma_space=10.0, radius_window_width=1):
    responses = []

    pool = Pool(processes=MAX_PROCESS_NUM)

    windows_width = radius_window_width
    total_window_length = 2 * windows_width + 1

    rows_every_workers = total_window_length // MAX_PROCESS_NUM
    start_row = -windows_width
    end_row = start_row + rows_every_workers

    data = input_image

    for r in range(0, MAX_PROCESS_NUM):
        args = (start_row, end_row, windows_width, r,
                data, sigma_space)
        res = pool.apply_async(run_gaussian_filter, args)

        responses.append(res)

        start_row += rows_every_workers

        if r == MAX_PROCESS_NUM - 2:
            end_row = windows_width + 1
        else:
            end_row += rows_every_workers

    for res in responses:
        res.wait()

    sum_fr = None

    for thread_id in range(0, MAX_PROCESS_NUM):

        if thread_id == 0:
            sum_fr = pickle.load(
                open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'gaussian_sum_fr{0}.tmp'.format(thread_id)), "rb"))
        else:
            sum_fr += pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'gaussian_sum_fr{0}.tmp'.format(thread_id)), "rb"))

        os.remove(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'gaussian_sum_fr{0}.tmp'.format(thread_id)))

    return sum_fr.clip(0.0, 255.0).astype(np.uint8)


def sobel(img):
    # 3x3 sobel kernel for the horizontal direction
    Mx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    # 3x3 sobel kernel for the vertical direction
    My = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)

    # Measure the gradient component in each orientation
    # ..., after convolving the Sobel kernel to the original image
    # Horizontal direction
    Gx = signal.convolve2d(img, Mx, boundary='symm', mode='same')

    # Vertical direction
    Gy = signal.convolve2d(img, My, boundary='symm', mode='same')

    # Find the absolute magnitude of the gradient at each pixel
    gradient_magnitude = np.sqrt(Gx * Gx + Gy * Gy)
    return gradient_magnitude
