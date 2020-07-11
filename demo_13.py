import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


#形态学转换
img = cv.imread('test_1.png', 0)
kernel = np.ones((5, 5), np.uint8)
#侵蚀
erosion = cv.erode(img, kernel, iterations=1)
plt.subplot(121), plt.imshow(img), plt.title('img')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(erosion), plt.title('erosion')
plt.xticks([]), plt.yticks([])
plt.show()

#扩张
dilation = cv.dilate(img, kernel, iterations=1)
#开运算,对消除噪声很有效
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
#闭运算,在关闭前景对象内部的小孔或对象上的小黑点时很有用
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
#形态学梯度,结果将看起来像对象的轮廓
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
#顶帽
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
#黑帽
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
cv.imshow('image', dilation)
cv.waitKey()
cv.imshow('image', opening)
cv.waitKey()
cv.imshow('image', closing)
cv.waitKey()
cv.imshow('image', gradient)
cv.waitKey()
cv.imshow('image', tophat)
cv.waitKey()
cv.imshow('image', blackhat)
cv.waitKey()
cv.destroyAllWindows()
