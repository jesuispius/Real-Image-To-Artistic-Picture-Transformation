import cv2
import streamlit as st
import numpy as np
from upload_image import upload_image
from filtering import GaussianFiltering, mean_filter, bilateral_filter, gaussian_filter, sobel_filter, MedianFiltering
from color_transfer import convert_color_space_RGB_to_GRAY
from scipy import signal
from testmediancut import k_mean


def sobel(img):
    kernel_horizontal = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float32)

    kernel_vertical = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=np.float32)

    Gx = signal.convolve2d(img, kernel_horizontal, boundary='symm', mode='same')
    Gy = signal.convolve2d(img, kernel_vertical, boundary='symm', mode='same')
    gradient_magnitude = np.sqrt(Gx * Gx + Gy * Gy)
    return gradient_magnitude


# Title of project
st.title("Real Image To Cartoon/Sketch Image Transformation")

# Upload image
original_img = upload_image(isShown=True)

if len(original_img) != 0:
    # Gray scale
    gray_img = convert_color_space_RGB_to_GRAY(original_img)
    gray_img = np.array(gray_img).astype(np.uint8)
    st.image(gray_img, "Gray scale")

    # Median blur
    img_median = MedianFiltering(gray_img, 5)
    img_median = img_median.astype(np.uint8)
    st.image(img_median, "Median Blurring with kernel (5x5)")

    # Sobel
    img_sobel = sobel(img_median)
    img_sobel = img_sobel.astype(np.uint8)
    img_sobel2 = cv2.bitwise_not(img_sobel)
    img_sobel2 = img_sobel2.astype(np.uint8)
    st.image(img_sobel, "Sobel Edge Detection")
    st.image(img_sobel2, "Sobel Edge Detection")

    # Remove noise
    img_bilateral = bilateral_filter(original_img, 30, 0.1, 1)
    img_bilateral = img_bilateral.astype(np.uint8)
    st.image(img_bilateral, "Bilateral Blurring")

    # Color quantization
    img_quantization = k_mean(img_bilateral)
    img_quantization = img_quantization.astype(np.uint8)
    st.image(img_quantization, "Color Quantization")

    final_img = cv2.bitwise_and(img_quantization, img_quantization, mask=img_sobel2)
    st.image(final_img)
