import cv2
import numpy as np

if __name__ == '__main__':

    img = cv2.imread("E:/download/Project/Program2/Src.jpg")

    height = img.shape[0]
    width = img.shape[1]

    new_img_b = np.zeros((height, width, 1), np.uint8)
    new_img_g = np.zeros((height, width, 1), np.uint8)
    new_img_r = np.zeros((height, width, 1), np.uint8)
    new_img_bgr = np.zeros((height, width, 1), np.uint8)

    for i in range(height):
        for j in range(width):
            b = img[i, j][0]
            g = img[i, j][1]
            r = img[i, j][2]
            new_img_b[i, j][0] = np.uint8(b)
            new_img_g[i, j][0] = np.uint8(g)
            new_img_r[i, j][0] = np.uint8(r)
            new_img_bgr[i,j][0] = 0.299*np.uint8(r) + 0.587*np.uint8(g) + 0.114*np.uint8(b)

    equalized = cv2.equalizeHist(new_img_bgr)

    blurred_1 = cv2.GaussianBlur(new_img_bgr, (0,0), 1)
    blurred_3 = cv2.GaussianBlur(new_img_bgr, (0,0), 3)
    blurred_5 = cv2.GaussianBlur(new_img_bgr, (0,0), 5)
    blurred_7 = cv2.GaussianBlur(new_img_bgr, (0,0), 7)

    cv2.imwrite("E:/download/Project/Program2/gray_b.jpg", new_img_b)
    cv2.imwrite("E:/download/Project/Program2/gray_g.jpg", new_img_g)
    cv2.imwrite("E:/download/Project/Program2/gray_r.jpg", new_img_r)
    cv2.imwrite("E:/download/Project/Program2/gray_bgr.jpg", new_img_bgr)
    cv2.imwrite("E:/download/Project/Program2/equalized.jpg", equalized)
    cv2.imwrite("E:/download/Project/Program2/blurred_1.jpg", blurred_1)
    cv2.imwrite("E:/download/Project/Program2/blurred_3.jpg", blurred_3)
    cv2.imwrite("E:/download/Project/Program2/blurred_5.jpg", blurred_5)
    cv2.imwrite("E:/download/Project/Program2/blurred_7.jpg", blurred_7)