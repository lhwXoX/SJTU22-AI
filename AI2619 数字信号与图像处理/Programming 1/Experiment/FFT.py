#This python code generate images of frequency domain
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
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return h

if __name__ == '__main__':
    gaussian = gaussian_2d((100, 100), 4)
    img = cv.imread('Antialiasing_sample.png', 0)
    spectrum_show(img, gaussian)

#参考:https://blog.csdn.net/weixin_45690354/article/details/120835594