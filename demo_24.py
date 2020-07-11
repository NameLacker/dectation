import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#二维直方图，图片用HSV表示(色调H， 饱和度S)
img = cv.imread('cat.jpg')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
#第二个参数因为是二维的所以有两个参数
hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv.imshow('hist', hist)
cv.waitKey()
cv.destroyAllWindows()
