import streamlit as st
import cv2
import numpy as np
from upload_image import upload_image

st.title("Real Image To Cartoon Image Transformation")

original_img = upload_image(isShown=True)

algo_option = st.selectbox(
    label='What algorithm would you like to choose?',
    options=('Median',
             'Gaussian',
             'Bilateral')
)
