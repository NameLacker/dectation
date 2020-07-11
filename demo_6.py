'''
图像的基本操作 
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('cat.jpg')

'''
px = img[100, 100]
print(px)

img[100][100] = [255, 255, 255]
print(img[100, 100])
'''

#print(img.item(100, 100, 2))
img.itemset((100, 100, 2), 100)
#print(img.item(100, 100, 2))

#print(img.shape)
#print(img.dtype)

cat_face = img[80:175, 170:310]
img[185:280, 0:140] = cat_face

#拆分和合并通道
#拆分通道
b, g, r = cv.split(img)
b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

print(b)
print(g)
print(r)

#合并通道
img = cv.merge((b, g, r))

#img[:, :, 2] = 0

cv.imshow('cat', img)
cv.waitKey()
cv.destroyAllWindows()