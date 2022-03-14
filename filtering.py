# ==================================================================================================================== #
# Collaborators: Phuoc Nguyen + Tri Le
# Description: This file includes the filters implementation for the entire of project
# Filename: filtering.py
# ==================================================================================================================== #

# Dependencies
from cmath import pi
import os
import numpy as np
import math
from multiprocessing import Pool
import pickle
import random
from scipy import ndimage
import string

from color_transfer import convert_color_space_RGB_to_GRAY


# Global Variable
TWO_PI = 2.0 * math.pi
MAX_THREAD = 6
MAX_PROCESS_NUM = 6


PI_1_d_8 = math.pi / 8
PI_1_d_4 = math.pi / 4
PI_1_d_2 = math.pi / 2
PI_3_d_4 = 3*math.pi / 4
PI_1_d_1 = math.pi


# -------------------------------------------------------------------------------------------------------------------- #
# MEDIAN FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def run_median_filter(file_name, start_row, end_row, im, kernel_size):
    """
    Median Filtering Function.

    :param file_name: filename
    :param start_row: start row
    :param end_row:
    :param im: image
    :param kernel_size: kernel size of filter
    :return: the image after applying median filter
    """
    # Initialisation
    edge = math.floor(kernel_size / 2)
    output_img = np.zeros_like(im[start_row:end_row, :])

    begin_row = start_row

    if begin_row == 0:
        begin_row = begin_row + edge

    stop_row = end_row

    if stop_row == im.shape[0]:
        stop_row = im.shape[0] - edge

    for x in range(begin_row, stop_row):
        for y in range(edge, im.shape[1] - edge):
            # Taking out the image matrix corresponding to the kernel size
            mask_img = im[x - edge: x + edge + 1, y - edge: y + edge + 1]

            # Sort the pixel in ascending order
            median = np.sort(np.ravel(mask_img))

            # Getting the median value from the sorted list
            median_val = int(kernel_size * kernel_size / 2)
            output_img[x - start_row, y] = median[median_val]

    pickle.dump(output_img, open(file_name, 'wb'), pickle.HIGHEST_PROTOCOL)


def median_filter(image, kernel_size=3):
    return run_parallel(image, run_median_filter, (kernel_size,))


# -------------------------------------------------------------------------------------------------------------------- #
# BILATERAL FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def run_bilateral_filter(start_col, end_col, window_width, thread_id, input_image, sigma_space, sigma_intensity):
    # def gaussian_kernel(data, sigma):
    #     return (1/(TWO_PI*sigma*sigma))*np.exp(-((data)/(2.0*sigma**2)))

    def our_kernel(data, sigma):
        return np.round(np.exp(-0.5*((data)/(sigma**2))))

    sum_fr = np.zeros(input_image.shape)
    sum_gs_fr = np.zeros(input_image.shape)  # input_image * EPSILON

    for w_col in range(start_col, end_col):
        for w_row in range(-window_width, window_width + 1):
            gs = our_kernel(w_col ** 2 + w_row ** 2, sigma_space)

            w_image = np.roll(input_image, [w_row, w_col], axis=[0, 1])

            fr = gs * our_kernel((w_image - input_image)
                                 ** 2, sigma_intensity)

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

    return (sum_gs_fr * 255.0).clip(0.0, 255.0)


# -------------------------------------------------------------------------------------------------------------------- #
# mean FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def run_mean_filter(start_col, end_col, window_width, thread_id, input_image):
    sum_fr = np.zeros(input_image.shape)

    for w_col in range(start_col, end_col):
        for w_row in range(-window_width, window_width + 1):
            w_image = np.roll(input_image, [w_row, w_col], axis=[0, 1])
            sum_fr += w_image

    pickle.dump(
        sum_fr,
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
             'mean_sum_fr{0}.tmp'.format(thread_id)), 'wb'),
        pickle.HIGHEST_PROTOCOL
    )


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

    return sum_fr.clip(0.0, 255.0)


def convolve(image, filtered):
    if len(image.shape) > 2:
        filter_expand = np.stack([filtered] * image.shape[2], axis=2)
    else:
        filter_expand = filtered

    return ndimage.filters.convolve(image, filter_expand)


# -------------------------------------------------------------------------------------------------------------------- #
# GAUSSIAN FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def gaussian_filters(img, kernel_size=3, sigma=1):
    g_kernel_matrix = gaussian_kernel(kernel_size, sigma)

    return convolve(img, g_kernel_matrix)


# -------------------------------------------------------------------------------------------------------------------- #
# SOBEL FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def sobel_filter(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    return G


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


# -------------------------------------------------------------------------------------------------------------------- #
# NON MAX SUPPRESSION
# -------------------------------------------------------------------------------------------------------------------- #
def non_max_suppression(img, gradient_angle):
    return run_parallel(img, run_non_max_suppression, (gradient_angle,))


def run_non_max_suppression(file_name, start_row, end_row, im, gradient_angle):
    rows = im.shape[0]
    cols = im.shape[1]

    result = np.zeros((end_row - start_row, cols), dtype=np.int32)

    begin_row = start_row
    if start_row == 0:
        begin_row = 1

    if end_row == rows:
        end_row = rows - 1

    gradient_angle[gradient_angle < 0] += PI_1_d_1

    run_angle = [(0, PI_1_d_8, (0, 1), (0, -1)),
                 (PI_1_d_4 - PI_1_d_8, PI_1_d_4 + PI_1_d_8, (1, -1), (-1, 1)),
                 (PI_1_d_2 - PI_1_d_8, PI_1_d_2 + PI_1_d_8, (1, 0), (-1, 0)),
                 (PI_3_d_4 - PI_1_d_8, PI_3_d_4 + PI_1_d_8, (-1, -1), (1, 1)),
                 (PI_1_d_1 - PI_1_d_8, PI_1_d_1, (0, 1), (0, -1))]

    for i in range(begin_row, end_row):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            for ra in run_angle:
                if ra[0] <= gradient_angle[i, j] < ra[1]:
                    q = im[i + ra[2][0], j + ra[2][1]]
                    r = im[i + ra[3][0], j + ra[3][1]]
                    break

            if (im[i, j] >= q) and (im[i, j] >= r):
                result[i - start_row, j] = im[i, j]
            else:
                result[i - start_row, j] = 0

    pickle.dump(result, open(file_name, 'wb'), pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------------------------------------- #
# THRESHOLD
# -------------------------------------------------------------------------------------------------------------------- #
def threshold(img, low_threshold=15, high_threshold=30):
    res = np.zeros_like(img, dtype=int)

    weak = 1
    strong = 255

    strong_x_cord, strong_y_cord = np.where(img >= high_threshold)

    weak_x_cord, weak_y_cord = np.where(
        (img < high_threshold) & (img >= low_threshold))

    res[strong_x_cord, strong_y_cord] = strong
    res[weak_x_cord, weak_y_cord] = weak

    return res, weak, strong


# -------------------------------------------------------------------------------------------------------------------- #
# HYSTERESIS
# -------------------------------------------------------------------------------------------------------------------- #
def hysteresis(img, weak=127, strong=255):
    return run_parallel(img, run_hysteresis, (weak, strong))


# Edge Tracking by Hysteresis
def run_hysteresis(file_name, start_row, end_row, im, weak=127, strong=255):
    rows = im.shape[0]
    cols = im.shape[1]

    begin_row = start_row
    if start_row == 0:
        begin_row == 1

    stop_row = end_row
    if end_row == 0:
        stop_row == rows - 1

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(begin_row, stop_row):
        for j in range(1, cols - 1):
            if im[i, j] == weak:
                for n in neighbors:
                    if (im[i + n[0], j + n[1]] == strong):
                        im[i, j] = strong
                        break
                if im[i, j] == weak:
                    im[i, j] = 0

    pickle.dump(im[start_row:end_row, :], open(
        file_name, 'wb'), pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------------------------------------- #
# CANNY EDGE DETECTION
# -------------------------------------------------------------------------------------------------------------------- #
def canny_edge_detection(img, gaussian_kernel=3, sigma=1):
    if len(img.shape) > 2:
        I = convert_color_space_RGB_to_GRAY(img)
    else:
        I = img

    # Noise reduce by gaussian
    I = gaussian_filters(I, gaussian_kernel, sigma)

    (I, gradient_angle) = sobel_filters(I)

    I = non_max_suppression(I, gradient_angle)

    (I, weak, strong) = threshold(I)

    I = hysteresis(I, weak, strong)

    return I


def run_parallel(im, func, parameters=()):
    def generate_file_name():
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for _ in range(10))

    rows = im.shape[0]

    process_result_list = []
    workers = Pool(processes=MAX_THREAD)

    rows_for_workers = rows // MAX_THREAD
    start_row = 0
    end_row = rows_for_workers

    list_files = []

    current_path = os.path.dirname(os.path.abspath(__file__))
    temp_folder = generate_file_name()
    current_path = os.path.join(current_path, temp_folder)
    os.mkdir(current_path)

    for process_num in range(0, MAX_THREAD):

        if end_row > start_row:
            file_name = generate_file_name()
            file_name = os.path.join(current_path, file_name)
            list_files.append(file_name)

            args = (file_name, start_row, end_row, im) + parameters
            process_result = workers.apply_async(func, args)
            process_result_list.append(process_result)

        start_row += rows_for_workers

        if process_num == MAX_THREAD - 2:
            end_row = rows
        else:
            end_row += rows_for_workers

    for process_result in process_result_list:
        process_result.wait()

    combine_image = []

    for file in list_files:
        data = pickle.load(open(file, "rb"))
        if data is not None:
            combine_image.append(data)
            os.remove(file)

    os.rmdir(current_path)
    return np.vstack(np.array(combine_image))


def color_reducer(im, num_of_level):
    return np.round(im / num_of_level)*num_of_level


def layer_separation(input_image, threshold):
    np.seterr(divide='ignore', invalid='ignore')
    def is_similar(point1, point2, threshold):

        a = point1
        b = point2
    
        diff = np.abs(100.0*(a - b)/a)

        if np.all(diff < threshold):
            return True

        return False

    image = input_image

    color_dict = {}
    total_color_dict = {}

    result = np.full(
        (input_image.shape[0], input_image.shape[1]), -1, dtype=int)
    result2 = np.full_like(input_image, 0, dtype=np.float64)

    stack_similar = [(0, 0)]

    stack_non_similar_dict = {}
    stack_non_similar = []

    label = 0

    color_dict[label] = input_image[0, 0]
    total_color_dict[label] = 1
    result[0, 0] = label

    while len(stack_similar) > 0 or len(stack_non_similar) > 0:

        if len(stack_similar) > 0:
            (x, y) = stack_similar.pop()

        elif len(stack_non_similar) > 0:

            (x, y) = stack_non_similar.pop()

            if np.any(result[x, y] == -1):
                label += 1
                result[x, y] = label
                color_dict[label] = input_image[x, y].copy()
                total_color_dict[label] = 1
            else:
                continue

        neighbors = [(x-1, y), (x+1, y), (x-1, y-1), (x+1, y+1),
                     (x-1, y+1), (x+1, y-1), (x, y-1), (x, y+1)]

        for n in neighbors:

            if (0 <= n[0] <= image.shape[0] - 1) and (0 <= n[1] <= image.shape[1]-1) and result[n[0], n[1]] == -1:

                if is_similar(image[x, y], image[n[0], n[1]], threshold):

                    result[n[0], n[1]] = label

                    color_dict[label] += input_image[x, y].copy()
                    total_color_dict[label] += 1

                    stack_similar.append(n)
                    if n in stack_non_similar_dict:
                        del stack_non_similar_dict[n]

                else:
                    if n not in stack_non_similar_dict:
                        stack_non_similar.append(n)
                        stack_non_similar_dict[n] = True

    color = {}
    for key in color_dict.keys():
        color[key] = color_dict[key] / total_color_dict[key]
    y = (input_image.shape[0]*input_image.shape[1]) // 100
    for r in range(result2.shape[0]):
        for c in range(result2.shape[1]):
            result2[r, c] = (input_image[r, c] + color[result[r, c]])/2

    return result2.clip(0.0, 255.0)


# if __name__ == '__main__':
#     file_name = "./img_test/image1.jpg"
#     img = cv2.imread(file_name)

#     start = timeit.default_timer()
#     new_img = sobel_filters(img)

#     cv2.imwrite('./img_test/result2.png', new_img)

#     stop = timeit.default_timer()
#     print('Time: ', stop - start)
