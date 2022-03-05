import numpy as np
import cv2


def adaptive_threshold(input_img):
    h, w, _ = input_img.shape
    S = w / 8
    s2 = S / 2
    T = 23.0

    # integral img
    int_img = np.zeros_like(input_img, dtype=np.uint32)
    # for col in range(w):
    #     for row in range(h):
    #         int_img[row, col] = input_img[0:row, 0:col].sum()
    for col in range(w):
        sumn = 0
        for row in range(h):
            sumn += sum(input_img[row, col])
            if col == 0:
                int_img[row, col] = sumn
            else:
                int_img[row, col] = input_img[row - 1, col] + sumn
    # output img
    out_img = np.zeros_like(input_img)

    for col in range(w):
        for row in range(h):
            # SxS region
            y0 = int(max(row - s2, 0))
            y1 = int(min(row + s2, h - 1))
            x0 = int(max(col - s2, 0))  # i
            x1 = int(min(col + s2, w - 1))  # i
            count = (y1 - y0) * (x1 - x0)
            sum_ = int_img[y1, x1] - int_img[y1, x0 - 1] - int_img[y0 - 1, x1] + int_img[y0 - 1, x0 - 1]
            value = input_img[row, col] * count <= sum_ * (100. - T) / 100.
            if value.all():
                out_img[row, col] = 0
            else:
                out_img[row, col] = 255
    cv2.imshow("image", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread("./testimage.jpg")
image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
print(image.shape)
adaptive_threshold(image)
