import streamlit as st
import cv2
import numpy as np
from upload_image import upload_image
from filtering import GaussianFiltering, MedianFiltering
from testmediancut import k_mean


# ============================================================================================================= #
def generateChoice(img):
    # Initialize the choice
    choice = 0

    # User chooses the option between generating or customizing to create user's cartoon/sketch image
    if len(img) != 0:
        st.write("Would you like to automatically generate a cartoon image or customize by yourself step by step?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate"):
                st.write("Generate your cartoon image!")
                choice = 1

        with col2:
            if st.button("Customize"):
                st.write("Customize your cartoon image!")
                choice = 2

    return choice


# ============================================================================================================= #
def generate_cartoon_image(img):
    with st.spinner('Please wait...'):
        # Convert original image to gray channel
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

        image = gray_img
        image = cv2.medianBlur(image, 5)
        edge_mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)

        # colour quantization
        # k value determines the number of colours in the image
        total_color = 8
        k = total_color

        # Transform the image
        data = np.float32(original_img).reshape((-1, 3))

        # Determine criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

        # Implementing K-Means
        ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(original_img.shape)
        st.image(result)
        st.image(k_mean(image))


# ============================================================================================================= #
# Title of project
st.title("Real Image To Cartoon/Sketch Image Transformation")

# Upload image
original_img = upload_image(isShown=True)

# Choose generate or customize the cartoon image
generating_choice = generateChoice(original_img)

if generating_choice == 1:
    generate_cartoon_image(img=original_img)
