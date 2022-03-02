import numpy as np
import math


def GaussianFunction(x, y, std_dev):
    return (1 / (2 * np.pi * std_dev ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * std_dev ** 2))


def GaussianFiltering(img, std_dev, kernel_size):
    h = img.shape[0]
    w = img.shape[1]

    img_pad = np.pad(
        img,
        (math.floor(kernel_size / 2), math.floor(kernel_size / 2)),
        'constant'
    )

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float)

    for c in range(0, kernel_size):
        for r in range(0, kernel_size):
            gx = c - math.floor(kernel_size / 2)
            gy = r - math.floor(kernel_size / 2)
            kernel[r, c] = GaussianFunction(gx, gy, std_dev)

    sum_kernel = kernel.sum()
    kernel = np.divide(kernel, sum_kernel)

    for row in range(0, h):
        for col in range(0, w):
            img[row, col] = np.sum(img_pad[row: row + kernel_size, col: col + kernel_size] * kernel)
    return img
