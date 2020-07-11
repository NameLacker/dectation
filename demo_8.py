'''
图像上的算术运算
'''

import numpy as np
import cv2 as cv

'''
#图像加法
x = np.uint8([250])
y = np.uint8([10])
#饱和运算
print(cv.add(x, y))
#模运算
print(x + y)
'''



'''
#图像融合
#G(x) = (1-a)*f_0(x) + a*f_1(x)
img1 = cv.imread('dog.jpg')
img2 = cv.imread('background.jpg')
print(img1.shape).
print(img2.shape)

img1 = img1[330:870, 480:1440]
print(img1.shape)

dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)
cv.imshow('dst', dst)
cv.waitKey()
cv.destroyAllWindows()
'''



#按位运算：AND, OR, NOT, XOR
img1 = cv.imread('dog.jpg')
img2 = cv.imread('logo.jpg')

t_1 = cv.getTickCount()

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

#转化为灰度图
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
#二值化
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
#按位取反
mask_inv = cv.bitwise_not(mask)
#将ROI中logo区域涂黑
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

img2_fg = cv.bitwise_and(img2, img2, mask=mask)

dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
cv.imshow('img', img1)
cv.waitKey()
cv.destroyAllWindows()

cv.imwrite('new_dog.jpg', img1)

t_2 = cv.getTickCount()
time = (t_2 - t_1)/cv.getTickFrequency()
print(time)
#print(img1.shape)
#print(img2.shape)



