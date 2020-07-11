import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Opencv中的轮廓
img = cv.imread('./images/000001.jpg')
#print(img.shape)
#转化为灰度图
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#print(img.shape)
#转化为二值图
ret, img2 = cv.threshold(img1, 127, 255, 0)
#print(img.shape)
#检测轮廓并绘制
image, contours, hierarchy = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

#轮廓的基本属性

#长宽比,weidth/height
x, y, w, h = cv.boundingRect(cnt)
aspect_retio = float(w)/h

#范围,范围是轮廓区域与边界矩形区域的比值。
area = cv.contourArea(cnt)
x, y, w, h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area

#坚实度,坚实度是等高线面积与其凸包面积之比
hull = cv.convexHull(cnt)
area_hull = cv.contourArea(hull)
area = cv.contourArea(cnt)
solidity = float(area)/area_hull

#等效直径,是与此面积相等的圆的直径
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)

#取向,是物体指向的角度
(x, y), (MA, ma), angle = cv.fitEllipse(cnt)
'''
#掩码和像素点
mask = np.zeros(img.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)

#最大值、最小值和它们的参数
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img2,mask = mask)

#平均颜色和平均强度
mean_val = cv.mean(img,mask = mask)
'''
#极端点
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
print(leftmost, rightmost, topmost, bottommost)