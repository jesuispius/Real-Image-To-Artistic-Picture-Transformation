import numpy as np


# Converting from BGR to RGB and vice-sera
def convert_color_space_BGR_to_RGB(img_BGR):
    """
    Function to convert from BGR color space to RGB color space.

    :param img_BGR: image in BGR color space
    :return: image in RGB color space
    """
    img_RGB = np.zeros_like(img_BGR, dtype=np.float32)
    img_RGB = img_BGR[:, :, ::-1]
    return img_RGB


def convert_color_space_RGB_to_BGR(img_RGB):
    """
    Function to convert from RGB color space to BGR color space.

    :param img_RGB: image in RGB color space
    :return: image in BGR color space
    """
    img_BGR = np.zeros_like(img_RGB, dtype=np.float32)
    img_BGR = img_RGB[:, :, ::-1]
    return img_BGR


def convert_color_space_RGB_to_GRAY(img_RGB):
    """
    Function to convert from RGB color space to GRAY channel.

    :param img_RGB: image in RGB color space
    :return: image in GRAY channel
    """
    weights = np.array([0.299, 0.587, 0.114])
    gray = np.dot(img_RGB[..., :3], weights)
    return gray
