import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''
#图像卷积
img = cv.imread('logo.jpg')
kernel = np.ones((5, 5), np.float32)/25
#-1表示输出图像与输入图像具备相同的深度
dst = cv.filter2D(img, -1, kernel)
#原图
plt.subplot(121), plt.imshow(img), plt.title('Orginal')
plt.xticks([]), plt.yticks([])
#滤波图
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
'''

'''
#平均模糊
img = cv.imread('logo.jpg')
#-1表示输出图像与输入图像具备相同的深度
dst = cv.blur(img, (5, 5))
#原图
plt.subplot(121), plt.imshow(img), plt.title('Orginal')
plt.xticks([]), plt.yticks([])
#滤波图
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
'''

'''
#高斯模糊
img = cv.imread('logo.jpg')
dst = cv.GaussianBlur(img, (5, 5), 0)
#原图
plt.subplot(121), plt.imshow(img), plt.title('Orginal')
plt.xticks([]), plt.yticks([])
#滤波图
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
'''

'''
#中位模糊
img = cv.imread('logo.jpg')
dst = cv.medianBlur(img, 5)
#原图
plt.subplot(121), plt.imshow(img), plt.title('Orginal')
plt.xticks([]), plt.yticks([])
#滤波图
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
'''

'''
#双边滤波
img = cv.imread('logo.jpg')
dst = cv.bilateralFilter(img, 9, 75, 75)
#原图
plt.subplot(121), plt.imshow(img), plt.title('Orginal')
plt.xticks([]), plt.yticks([])
#滤波图
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
'''