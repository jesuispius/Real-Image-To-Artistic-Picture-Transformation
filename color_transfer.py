import numpy as np


# Converting from BGR to RGB and vice-sera
def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR, dtype=np.float32)
    img_RGB = img_BGR[:, :, ::-1]
    return img_RGB


def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB, dtype=np.float32)
    img_BGR = img_RGB[:, :, ::-1]
    return img_BGR
