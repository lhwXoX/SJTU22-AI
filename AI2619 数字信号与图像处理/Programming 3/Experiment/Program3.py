import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    num_samples_per_class = 500
    labels_cnt = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    print(labels_cnt/255)
    img = cv2.imread('./Figure/Koishi.jpg', cv2.IMREAD_GRAYSCALE)
    height = img.shape[0]
    width = img.shape[1]
    num = 25
    dist = int(width/num)

    img_period = np.zeros((height, width), np.uint8)

    for i in range(num+1):
        img_period[:,dist*i:dist*i+3] = 200

    img_combine = cv2.add(img, img_period)
    
    f_1st = np.fft.fft2(img)
    f_1st = np.fft.fftshift(f_1st)
    
    crow, ccol = height  // 2, width // 2
    mask = np.zeros((height, width), int)
    r = 6
    r2 = 46
    center = [crow, ccol]
    x, y = np.ogrid[:height, :width]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r
    mask[mask_area] = 1
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r2*r2
    mask[mask_area] = 0
    img_filterd = f_1st * mask

    f_2st = np.fft.fft2(img_filterd)
    plt.imshow(np.log(1 + np.abs(f_2st/3e7)), cmap='gray')
    plt.show()
    '''
    cv2.imwrite("E:/download/Project/Program3/Koishi_gray.jpg", img)
    cv2.imwrite("E:/download/Project/Program3/grating.jpg", img_period)
    cv2.imwrite("E:/download/Project/Program3/Koishi_combined.jpg", img_combine)
    '''
    
    
    