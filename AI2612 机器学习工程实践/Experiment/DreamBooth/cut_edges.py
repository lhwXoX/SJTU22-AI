import cv2
import numpy as np
import os



def remove_border(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Canny 边缘检测
    edges = cv2.Canny(gray, 40, 100)

    # 形态学操作：闭运算以填补小孔
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 找到轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # 获取矩形边界框
        x, y, w, h = cv2.boundingRect(max_contour)
        # 裁剪图像
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image
    print("No contours found.")
    return image  # 如果没有找到轮廓，返回原图


def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 处理常见图片格式
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            cropped_image = remove_border(image)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)


path = '/Users/wangruihan/Documents/本科课程/机器学习/project/data'
input_folder = '/poster'
output_folder = '/poster_1'
process_images_in_folder(path+input_folder, path+output_folder)
