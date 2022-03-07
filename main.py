import streamlit as st
import cv2
import numpy as np
from PIL import Image
from upload_image import upload_image
from filtering import GaussianFiltering, mean_filter, bilateral_filter, gaussian_filter,sobel_filter, MedianFiltering
from color_transfer import convert_color_space_RGB_to_GRAY


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
    st.image(img_median, "Median Blur")

    # Sobel
    img_sobel = sobel_filter(original_img)
    st.image(img_sobel)
