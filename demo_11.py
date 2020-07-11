import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('bin.png', 0)

'''
#简单阈值
#pixel>threshold 置255 反之置0
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#pixel>threshold 置0 反之置255
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
#pixel>threshold 置maxval 反之不变
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
#pixel>threshold 不变 反之置0
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
#pixel>threshold 置0 反之不变
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

#标题
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#图片
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
'''

#自适应阈值
#全局阈值
ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#Otsu阈值
ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#高斯滤波后再采用Otsu阈值
blur = cv.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

#绘制所有图像及其直方图
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()