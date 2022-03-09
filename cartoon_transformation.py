import streamlit as st
import cv2
import numpy as np
from upload_image import upload_image
from filtering import mean_filter, bilateral_filter, median_filter, gaussian_filters,sobel_filters


# ============================================================================================================= #
def generateChoice(img):
    # Initialize the choice
    choice = 0

    # User chooses the option between generating or customizing to create user's cartoon/sketch image
    if len(img) != 0:
        st.write(
            "Would you like to automatically generate a cartoon image or customize by yourself step by step?")

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
        image = sobel_filters(image)
        # edge_mask = cv2.adaptiveThreshold(
        #     image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        # edge_mask = sobel_filter(image)

        # # colour quantization
        # # k value determines the number of colours in the image
        # total_color = 8
        # k = total_color

        # # Transform the image
        # data = np.float32(original_img).reshape((-1, 3))

        # # Determine criteria
        # criteria = (cv2.TERM_CRITERIA_EPS +
        #             cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

        # # Implementing K-Means
        # ret, label, center = cv2.kmeans(
        #     data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # center = np.uint8(center)
        # result = center[label.flatten()]
        # result = result.reshape(original_img.shape)

        # color = bilateral_filter(result, 30, 0.1, 2)
        # color = cv2.detailEnhance(color, sigma_s=10, sigma_r=0.15)
        # cartoon = cv2.bitwise_and(color, color, mask=edge_mask)
        st.image(image)


def customize_cartoon_image(img):
    # Convert original image to gray channel
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # Select the algorithm to style the image
    algo_option = st.selectbox(
        label='What algorithm would you like to choose?',
        options=('Median',
                 'Gaussian',
                 'Bilateral')
    )

    image_used = gray_img
    if algo_option == 'Median':
        with st.spinner('Please wait...'):
            image_used = mean_filter(image_used, 5)
            st.image(image_used, caption='Median Filtered Image')

    elif algo_option == 'Gaussian':
        with st.spinner('Please wait...'):
            image_used = gaussian_filter(image_used, 3, 3)
            st.image(image_used, caption='Gaussian Filtered Image')
    else:
        with st.spinner('Please wait...'):
            image_used = bilateral_filter(image_used, 30, 0.1, 1)
            st.image(image_used, caption='Bilateral Filtered Image')

    # Using the edge
    edge_option = st.selectbox(
        label='Do you want to increase the edge thickness?',
        options=('Yes',
                 'No')
    )

    if edge_option == 'Yes':
        adaptive_img = cv2.adaptiveThreshold(
            image_used,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            7,
            7
        )
        # color = cv2.bilateralFilter(image_used, 9, 250, 250)
        # color = cv2.detailEnhance(image_used, sigma_s=10, sigma_r=0.15)
        # cartoon = cv2.bitwise_and(color, color, mask=adaptive_img)
        # st.image(cartoon, caption='Adaptive Edge Mask')

        # colour quantization
        # k value determines the number of colours in the image
        total_color = 8
        k = total_color

        # Transform the image
        data = np.float32(original_img).reshape((-1, 3))

        # Determine criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

        # Implementing K-Means
        ret, label, center = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(original_img.shape)

        st.image(result)


# ============================================================================================================= #
# Title of project
st.title("Real Image To Cartoon/Sketch Image Transformation")

# Upload image
original_img = upload_image(isShown=True)

# Choose generate or customize the cartoon image
generating_choice = generateChoice(original_img)

if generating_choice == 1:
    generate_cartoon_image(img=original_img)
elif generating_choice == 2:
    customize_cartoon_image(img=original_img)
