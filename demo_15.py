import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Canny边缘检测
img = cv.imread('12.jpg', 0)
canny = cv.Canny(img, 100, 200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
cv.imwrite('output.jpg', canny)