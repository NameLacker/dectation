import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('star.png',0)
img2 = cv.imread('star2.png',0)
ret, thresh = cv.threshold(img1, 127, 255,0)
ret, thresh2 = cv.threshold(img2, 127, 255,0)
ret, contours,hierarchy = cv.findContours(thresh,2,1)
cnt1 = contours[0]
ret, contours,hierarchy = cv.findContours(thresh2,2,1)
cnt2 = contours[0]

# 比较两个形状或两个轮廓，并返回一个显示相似性的度量。
# 结果越低，匹配越好。它是根据矩值计算出来的。
ret = cv.matchShapes(cnt1,cnt2,1,0.0)
print(ret)

