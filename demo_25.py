import numpy as np
import cv2 as cv
roi = cv.imread('rose.jpg')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
target = cv.imread('dog.jpg')
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# 计算对象的直方图
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# 直方图归一化并利用反传算法
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# 用圆盘进行卷积
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)
# 应用阈值作与操作
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
res = np.vstack((target,thresh,res))
cv.imwrite('res.jpg',res)