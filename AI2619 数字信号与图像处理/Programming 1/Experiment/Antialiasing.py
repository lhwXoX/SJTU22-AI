#This python code includes the whole process: blur, sample and interpolation.
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('Antialiasing_sample.png')

#获取图像高度和宽度
height = img.shape[0]
width = img.shape[1]
afterheight = 80
afterwidth = 80

#进行高斯滤波
blurred_img = cv2.GaussianBlur(img, (0,0), 4)

#采样转换成16*16区域
numHeight = int(height/afterheight)
numwidth = int(width/afterwidth)

#创建一幅图像
new_img = np.zeros((afterheight, afterwidth, 3), np.uint8)

#图像循环采样16*16区域
for i in range(afterheight):
    #获取Y坐标
    y = i*numHeight
    y = int(y)
    for j in range(afterwidth):
        #获取X坐标
        x = j*numwidth
        x = int(x)
        #获取填充颜色 左上角像素点
        b = blurred_img[y, x][0]
        g = blurred_img[y, x][1]
        r = blurred_img[y, x][2]
        #填充到新图像中
        new_img[i, j][0] = np.uint8(b)
        new_img[i, j][1] = np.uint8(g)
        new_img[i, j][2] = np.uint8(r)
        
#显示原图像和采样后图像
cv2.imshow("source", img)
cv2.imshow("blurred", blurred_img)

#进行插值
fx = height/afterheight   #列数变回原来  
fy = width/afterwidth   #行数变回原来
resized = cv2.resize(new_img, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)   #线性插值

cv2.imshow("INTER_LINEAR image", resized)

#保存图像
cv2.imwrite("E:/download/Microsoft VS Code/project/Antialiasing_outputPic/80_Gaussianblurred_without_sample.jpg", blurred_img)
cv2.imwrite("E:/download/Microsoft VS Code/project/Antialiasing_outputPic/80_Gaussianblurred_then_sample_LINEAR.jpg", resized)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Reference:
https://cloud.tencent.com/developer/article/2321543
https://cloud.tencent.com/developer/article/1620628
'''