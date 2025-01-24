#I put the other two python files all together to generate images of frequency domains of different images.
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import skimage
 
def spectrum_show(img1, img2):         # 定义一个用于计算灰度图的频谱图并显示的函数
 f = np.fft.fft2(img1)           # 快速傅里叶变换算法得到频率分布
 fshift = np.fft.fftshift(f)    # 将图像中的低频部分移动到图像的中心，默认是在左上角
 fimg = np.log(np.abs(fshift))  # fft结果是复数, 其绝对值结果是振幅，取对数的目的是将数据变换到0~255

 f2 = np.fft.fft2(img2)           
 fshift2 = np.fft.fftshift(f2)    
 fimg2 = np.log(np.abs(fshift2))
 # 展示结果
 plt.subplot(2,2,1), plt.imshow(img1, 'gray'), plt.title('Original Fourier')
 plt.axis('off')
 plt.subplot(2,2,2), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
 plt.axis('off')
 plt.subplot(2,2,3), plt.imshow(img2, 'gray'), plt.title('Original Fourier')
 plt.axis('off')
 plt.subplot(2,2,4), plt.imshow(fimg2, 'gray'), plt.title('Fourier Fourier')
 plt.axis('off')
 plt.show()
 
def gaussian_2d(shape, sigma):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) ) #高斯函数表达式
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return h

if __name__ == '__main__':
    gaussian = gaussian_2d((100, 100), 4)
    img = cv.imread('Antialiasing_sample.png', 0)

    height = img.shape[0]
    width = img.shape[1]
    afterheight = 80
    afterwidth = 80

    numHeight = int(height/afterheight)
    numwidth = int(width/afterwidth)
    new_img = np.zeros((afterheight, afterwidth, 1))
    for i in range(afterheight):
    #获取Y坐标
        y = i*numHeight
        y = int(y)
        for j in range(afterwidth):
            #获取X坐标
            x = j*numwidth
            x = int(x)
            #获取填充颜色 左上角像素点
            b = img[y, x]
            #填充到新图像中
            new_img[i, j] = np.uint8(b)

    fx = height/afterheight   #列数变回原来  
    fy = width/afterwidth   #行数变回原来
    resized = cv.resize(new_img, dsize=None, fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)

    spectrum_show(img, resized)#获得原图、直接采样插值后的图的fft

    blurred_img = cv.GaussianBlur(img, (0,0), 4)
    new_img2 = np.zeros((afterheight, afterwidth, 1))

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
            b = blurred_img[y, x]
        #填充到新图像中
            new_img2[i, j] = np.uint8(b)

    fx = height/afterheight   #列数变回原来  
    fy = width/afterwidth   #行数变回原来
    resized2 = cv.resize(new_img2, dsize=None, fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)

    spectrum_show(img, blurred_img)#获得原图、模糊后的fft
    spectrum_show(img, resized2)#获得原图、模糊后采样插值后的fft
    spectrum_show(resized, resized2)#获得有无模糊化的两张采样插值得到的图的频域