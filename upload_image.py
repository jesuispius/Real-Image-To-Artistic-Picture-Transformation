import streamlit as st
import cv2
from PIL import Image
import numpy as np


def upload_image(isShown=False):
    """
    Function to upload the image from computer/device

    :param isShown: Whether showing the image on the web page or not. Default: False
    :return: original image under numpy format
    """
    # Set original image to None
    original_img = []

    # Image extension file allowed
    extension_allowed = ['png', 'jpg', 'jpeg', 'svg']

    # Upload a file from user's computer/device
    uploaded_img = st.file_uploader(
        label="Upload Image",
        type=extension_allowed,
        accept_multiple_files=False,
    )

    # Image uploaded already?
    if uploaded_img is not None:
        # Open uploaded image
        pil_image = Image.open(uploaded_img)

        # Show the image in the web page
        if isShown:
            st.subheader("Original Image:")
            st.image(image=pil_image, caption="Original Image")

        # Convert image into numpy
        original_img = np.array(pil_image)

    return original_img
