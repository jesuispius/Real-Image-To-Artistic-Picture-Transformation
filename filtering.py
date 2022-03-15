# ==================================================================================================================== #
# Collaborators: Phuoc Nguyen + Tri Le
# Description: This file includes the filters implementation for the entire of project
# Filename: filtering.py
# ==================================================================================================================== #

# Dependencies
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
MAX_PROCESS_NUM = 1
LOG_CONST = math.log(0.5)
WEAK_POINT = 1
STRONG_POINT = 255

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
    def our_kernel1(data, sigma):
        '''
        This function is for alterternating gaussian kernel and improving the bilateral filter
        This function is used for scalar inputs
        '''
        return 1.0 if -0.5*(data)/(sigma**2) >= LOG_CONST else 0.0

    def our_kernel2(data, sigma):
        '''
        This function is for alterternating gaussian kernel and improving the bilateral filter
        This function is used for array inputs
        '''
        return (-0.5*(data)/(sigma**2) >= LOG_CONST).astype(int)

    # Initialize variables
    sum_weights = np.zeros(input_image.shape)
    sum_intensities = np.zeros(input_image.shape)

    # for each pixel in the kernel windows
    for w_col in range(start_col, end_col):
        for w_row in range(-window_width, window_width + 1):

            # calculate weights for space
            gs = our_kernel1(w_col ** 2 + w_row ** 2, sigma_space)

            # For each pixel P  in the image
            intensity = np.roll(input_image, [w_row, w_col], axis=[0, 1])

            weight = gs * our_kernel2((intensity - input_image)
                                      ** 2, sigma_intensity)

            sum_intensities += intensity * weight
            sum_weights += weight

    # return result in files
    pickle.dump(sum_weights, open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bilateralsum_fr{0}.tmp'.format(thread_id)), 'wb'),
        pickle.HIGHEST_PROTOCOL)

    pickle.dump(sum_intensities, open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bilateralsum_gs_fr{0}.tmp'.format(thread_id)), 'wb'),
        pickle.HIGHEST_PROTOCOL)


def bilateral_filter(input_image, sigma_space=10.0, sigma_intensity=0.1, radius_window_width=1):
    responses = []

    pool = Pool(processes=MAX_THREAD)

    windows_width = radius_window_width
    total_window_length = 2 * windows_width + 1

    rows_every_workers = total_window_length // MAX_THREAD
    start_row = -windows_width
    end_row = start_row + rows_every_workers

    data = input_image.astype(np.float32) / 255.0

    # assign tasks for works
    for r in range(0, MAX_THREAD):

        # arguments of the function
        args = (start_row, end_row, windows_width, r,
                data, sigma_space, sigma_intensity)

        # call the function
        res = pool.apply_async(run_bilateral_filter, args)

        responses.append(res)
        start_row += rows_every_workers

        if r == MAX_THREAD - 2:
            end_row = windows_width + 1
        else:
            end_row += rows_every_workers

    # wait until the works done
    for res in responses:
        res.wait()

    sum_fr = None
    sum_gs_fr = None

    # combine the result
    for thread_id in range(0, MAX_THREAD):

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

    # return the result
    sum_gs_fr = sum_gs_fr / sum_fr

    return (sum_gs_fr * 255.0).clip(0.0, 255.0)


# -------------------------------------------------------------------------------------------------------------------- #
# convolve
# -------------------------------------------------------------------------------------------------------------------- #
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
    '''
    Generate the gaussian kernel matrix
    '''
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def gaussian_filters(img, kernel_size=3, sigma=1):
    '''
    Gaussian filter
    param kernel_size: the size of the kernel matrix
    sigma: the size of the blur circle.
    '''
    g_kernel_matrix = gaussian_kernel(kernel_size, sigma)
    return convolve(img, g_kernel_matrix)


# -------------------------------------------------------------------------------------------------------------------- #
# SOBEL FILTERING
# -------------------------------------------------------------------------------------------------------------------- #
def sobel_filter(img):

    convolve_matrix_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    convolve_matrix_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Gx = convolve(img, convolve_matrix_x)
    Gy = convolve(img, convolve_matrix_y)

    #calculate the magnitude of the gradient
    G = np.hypot(Gx, Gy)
    
    #normalize data
    G = G / G.max() * 255 

    return G

# -------------------------------------------------------------------------------------------------------------------- #
# SOBEL FILTERING and GRADIENT ANGLE
# -------------------------------------------------------------------------------------------------------------------- #
def sobel_filters_and_gradient_angle(img):

    convolve_matrix_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    convolve_matrix_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Gx = convolve(img, convolve_matrix_x)
    Gy = convolve(img, convolve_matrix_y)

    #calculate the magnitude of the gradient
    G = np.hypot(Gx, Gy)
    
    #normalize data
    G = G / G.max() * 255 

    #calculate gradient angles
    theta = np.arctan2(Gy, Gx)

    return G, theta


# -------------------------------------------------------------------------------------------------------------------- #
# NON MAX SUPPRESSION
# -------------------------------------------------------------------------------------------------------------------- #
def non_max_suppression(img, gradient_angle):
    '''
    Purpose: make the edges thinner
    This function can run parallely 
    '''
    return run_parallel(img, run_non_max_suppression, (gradient_angle,))


def run_non_max_suppression(file_name, start_row, end_row, im, gradient_angle):
    rows = im.shape[0]
    cols = im.shape[1]

    #initialize result
    result = np.zeros((end_row - start_row, cols), dtype=np.int32)

    begin_row = start_row
    if start_row == 0:
        begin_row = 1

    if end_row == rows:
        end_row = rows - 1

    gradient_angle[gradient_angle < 0] += PI_1_d_1

    #contain all case of angles and postion of r and q
    run_angle = [(0, PI_1_d_8, (0, 1), (0, -1)),
                 (PI_1_d_4 - PI_1_d_8, PI_1_d_4 + PI_1_d_8, (1, -1), (-1, 1)),
                 (PI_1_d_2 - PI_1_d_8, PI_1_d_2 + PI_1_d_8, (1, 0), (-1, 0)),
                 (PI_3_d_4 - PI_1_d_8, PI_3_d_4 + PI_1_d_8, (-1, -1), (1, 1)),
                 (PI_1_d_1 - PI_1_d_8, PI_1_d_1, (0, 1), (0, -1))]

    for i in range(begin_row, end_row):
        for j in range(1, cols - 1):
            q = 255
            r = 255
            #find postion of q and r based on the angle
            for ra in run_angle:
                if ra[0] <= gradient_angle[i, j] < ra[1]:
                    q = im[i + ra[2][0], j + ra[2][1]]
                    r = im[i + ra[3][0], j + ra[3][1]]
                    break
            
            #narrow the edge 
            if (im[i, j] >= q) and (im[i, j] >= r):
                result[i - start_row, j] = im[i, j]
            else:
                result[i - start_row, j] = 0
    
    #return result
    pickle.dump(result, open(file_name, 'wb'), pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------------------------------------- #
# THRESHOLD
# -------------------------------------------------------------------------------------------------------------------- #
def threshold(img, low_threshold=15, high_threshold=30):
    '''
    This function threshold the image
    Return the array of threshold
    '''

    result = np.zeros_like(img, dtype=int)

    strong_x_cord, strong_y_cord = np.where(img >= high_threshold)

    weak_x_cord, weak_y_cord = np.where(
        (img < high_threshold) & (img >= low_threshold))

    result[strong_x_cord, strong_y_cord] = STRONG_POINT
    result[weak_x_cord, weak_y_cord] = WEAK_POINT

    return result


# -------------------------------------------------------------------------------------------------------------------- #
# HYSTERESIS
# -------------------------------------------------------------------------------------------------------------------- #
def hysteresis(img):
    '''
    This function is for connecting weak point to strong point
    param: img is the input image
    '''
    return run_parallel(img, run_hysteresis, ())


# Edge Tracking by Hysteresis
def run_hysteresis(file_name, start_row, end_row, im):
    rows = im.shape[0]
    cols = im.shape[1]

    begin_row = start_row
    if start_row == 0:
        begin_row == 1

    stop_row = end_row
    if end_row == 0:
        stop_row == rows - 1
    
    # define the neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]

    
    for i in range(begin_row, stop_row):
        for j in range(1, cols - 1):
            #if we see a week point
            if im[i, j] == WEAK_POINT:

                #check all neighbors of that week point
                for n in neighbors:
                    
                    # there exists a strong points in the neighbors, 
                    # the turn the week point to a strong point 
                    if (im[i + n[0], j + n[1]] == STRONG_POINT):
                        im[i, j] = STRONG_POINT
                        break
                
                #if there no strong point in this week point,
                #then remove it
                if im[i, j] == WEAK_POINT:
                    im[i, j] = 0

    #return result via file
    pickle.dump(im[start_row:end_row, :], open(
        file_name, 'wb'), pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------------------------------------- #
# CANNY EDGE DETECTION
# -------------------------------------------------------------------------------------------------------------------- #
def canny_edge_detection(img, gaussian_kernel=3, sigma=1):
    '''
        edge detection using Canny algrothrim
    '''
    if len(img.shape) > 2:
        I = convert_color_space_RGB_to_GRAY(img)
    else:
        I = img

    # Noise reduce by gaussian
    I = gaussian_filters(I, gaussian_kernel, sigma)

    # Apple Sobel filter
    (I, gradient_angle) = sobel_filters_and_gradient_angle(I)

    # Make edges thinner 
    I = non_max_suppression(I, gradient_angle)

    # set threshold
    I = threshold(I)

    #Edge tracking and linking
    I = hysteresis(I)

    #return result
    return I


def run_parallel(im, func, parameters=()):
    '''
    Help function run parallely
    '''
    def generate_file_name():
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for _ in range(10))


    rows = im.shape[0]

    process_result_list = []
    workers = Pool(processes=MAX_THREAD)

    #initial scope for every workers
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
            #create file name for returning results
            file_name = generate_file_name()
            file_name = os.path.join(current_path, file_name)
            list_files.append(file_name)

            #call the function
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
    #reduce color in a image into some level of color
    num_color_in_level = 255 // num_of_level
    level = im // num_color_in_level
    return level * num_color_in_level

