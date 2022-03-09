# ==================================================================================================================== #
# Collaborators: Phuoc Nguyen + Tri Le
# Description: Main user interface using streamlit framework
# Filename: project_final.py
# ==================================================================================================================== #

# Dependencies
import streamlit as st
import cv2
import numpy as np
from upload_image import upload_image, read_image
from filtering import bilateral_filter, MedianFiltering, sobel
from color_transfer import convert_color_space_RGB_to_GRAY, convert_color_space_RGB_to_BGR
from testmediancut import k_mean


@st.cache(suppress_st_warning=True)
def download_image(filename):
    with open(filename, "rb") as file:
        st.download_button(
            label="Download image",
            data=file,
            file_name=filename,
            mime="image/png"
        )


def save_and_display_image(img, label, path, is_shown=False, is_downloaded=False):
    image = np.array(img).astype(np.uint8)

    if is_shown:
        st.image(image, label)

    filename = path + label.strip() + ".png"

    if is_downloaded:
        image = convert_color_space_RGB_to_BGR(image)
        image = image.astype(np.uint8)
        cv2.imwrite(filename, image)
        download_image(filename)
    else:
        cv2.imwrite(filename, image)

    return image


def generate_image(img_RGB, path, is_shown=False):
    if len(img_RGB) != 0:
        # Gray scale
        with st.spinner('Please waiting...'):
            gray_img = convert_color_space_RGB_to_GRAY(img_RGB)
        gray_img = save_and_display_image(gray_img, "Gray scale", path, is_shown, False)

        # Median blur
        with st.spinner('Please waiting...'):
            img_median = MedianFiltering(gray_img, 5)
        img_median = save_and_display_image(img_median, "Median Blurring with kernel (5x5)", path, is_shown, False)

        # Sobel
        with st.spinner('Please waiting...'):
            img_sobel = sobel(img_median)
        img_sobel = save_and_display_image(img_sobel, "Sobel Edge Detection 1", path, is_shown, False)

        with st.spinner('Please waiting...'):
            img_sobel2 = cv2.bitwise_not(img_sobel)
        img_sobel2 = save_and_display_image(img_sobel2, "Sobel Edge Detection 2", path, is_shown, False)

        # Remove noise
        with st.spinner('Please waiting...'):
            img_bilateral = bilateral_filter(img_RGB, 30, 0.1, 1)
        img_bilateral = save_and_display_image(img_bilateral, "Bilateral Blurring", path, is_shown, False)

        # Color quantization
        with st.spinner('Please waiting...'):
            img_quantization = k_mean(img_bilateral)
        img_quantization = save_and_display_image(img_quantization, "Color Quantization", path, is_shown, False)

        with st.spinner('Please waiting...'):
            final_img = cv2.bitwise_and(img_quantization, img_quantization, mask=img_sobel2)
        final_img = save_and_display_image(final_img, "Color Quantization", path, True, True)
        return final_img
    return np.array(([]))


def auto_generate_image(img_RGB):
    final_img = generate_image(img_RGB, path="./img/result/auto/", is_shown=False)
    return final_img


def generate_example_image(img_RGB):
    final_img = generate_image(img_RGB, path="./img/result/example/", is_shown=True)
    return final_img


# Title of project
st.title("Real Image To Artistic Picture Transformation")

section = st.sidebar.selectbox(
    label="Choose a section:",
    options=[
        "",
        "Example",
        "Auto Generate",
        "Customize"
    ]
)

if section == "Example":
    # Upload image
    original_img = read_image("./img/src/christopher.jpg")
    generate_example_image(original_img)
    st.stop()

elif section == "Auto Generate":
    # Upload image
    original_img = upload_image(isShown=True)
    auto_generate_image(original_img)
    st.stop()
