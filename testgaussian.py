import cv2
import numpy as np
from gaussian_filtering import GaussianFiltering


def rmse(apath, bpath):
    """
    This is the help function to get RMSE score.
    apath: path to your result
    bpath: path to our reference image
    when saving your result to disk, please clip it to 0,255:
    .clip(0.0, 255.0).astype(np.uint8))
    """
    a = cv2.imread(apath).astype(np.float32)
    b = cv2.imread(bpath).astype(np.float32)
    print(np.sqrt(np.mean((a - b) ** 2)))


original = cv2.imread('./testimage.jpg', 1)
original_img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
gau_img = GaussianFiltering(gray_img, 3, 3)

gau_img1 = cv2.GaussianBlur(gray_img, (3, 3), 3)
med_img = gau_img.clip(0.0, 255.0).astype(np.uint8)
cv2.imwrite('./result/ret_image_written.png', gau_img)
cv2.imwrite('./result/ret_image_cv2.png', gau_img1)
rmse('./result/ret_image_written.png', './result/ret_image_cv2.png')
