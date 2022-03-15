# ==================================================================================================================== #
# Collaborators: Phuoc Nguyen + Tri Le
# Description: Main user interface using streamlit framework
# Filename: project_final.py
# ==================================================================================================================== #

# Dependencies
import streamlit as st
import cv2
import numpy as np
from upload_image import upload_image
from color_transfer import convert_color_space_RGB_to_BGR
from our_algorithm import landscape, portrait


def save_and_display_image(img, label, path, is_shown=False):
    """
    Function to save and display the image on the webpage.

    :param img: image
    :param label: label of image
    :param path: where to store image
    :param is_shown: whether showing image on the webpage or not
    :return: image result
    """
    image = np.array(img).astype(np.uint8)

    if is_shown:
        st.image(image, label)

    filename = path + label.strip() + ".png"

    image = convert_color_space_RGB_to_BGR(image)
    image = image.astype(np.uint8)
    cv2.imwrite(filename, image)

    return image


def gen_landscape_image(img, path):
    """
    Function to generate the landscape image.

    :param img: original image
    :param path: path
    :return: image after filtering
    """
    with st.spinner('Please waiting...'):
        img_landscape = landscape(img)
    img_landscape = save_and_display_image(
        img_landscape, "Landscape Image", path, True)
    return img_landscape


def gen_portrait_image(img, path):
    """
    Function to generate the portrait image.

    :param img: original image
    :param path: path
    :return:
    """
    with st.spinner('Please waiting...'):
        img_portrait = portrait(img)
    img_portrait = save_and_display_image(
        img_portrait, "Portrait Image", path, True)
    return img_portrait


def warning_notifications():
    st.warning('Notice: Streamlit is quite weird when change the tab and when we choose another picture to upload. '
               'It will refresh all current tasks, so it take a little bit longer to redo the task. '
               'To avoid this problem, as you choose another image to upload or change to another tab, please '
               'make sure you remove the current original image right below the upload file section before making '
               'another actions. Thank you so much!')


# ==================================================================================================================== #
# USER INTERFACE
# ==================================================================================================================== #
st.title("Real Image To Artistic Picture")
st.sidebar.header("Welcome, User!")
section = st.sidebar.selectbox(
    label="Choose a section:",
    options=[
        "About",
        "Landscape",
        "Portrait"
    ]
)

if section == "About":
    st.markdown('### :memo: Description: ')
    st.markdown('Our application is built to generate an artistic picture from a real image.')
    st.markdown('### :handshake: Author: ')
    st.markdown(':two_women_holding_hands: Collaborators: '
                '[Phuoc Nguyen](https://github.com/jesuispius)'
                ' and [Tri Le](https://github.com/trilq142)')

elif section == "Landscape":
    # Sub header
    st.header("Artistic Landscape Picture Transformation")

    # Upload image
    original_img = upload_image(isShown=True)

    # Image processing
    if len(original_img) != 0:
        st.subheader("Result: ")
        gen_landscape_image(original_img, path="./img/result/landscape/")
        st.stop()

    warning_notifications()

elif section == "Portrait":
    # Sub header
    st.header("Artistic Portrait Picture Transformation")

    # Upload image
    original_img = upload_image(isShown=True)

    # Image processing
    if len(original_img) != 0:
        st.subheader("Result: ")
        gen_portrait_image(original_img, path="./img/result/portrait/")
        st.stop()

    warning_notifications()
