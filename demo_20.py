import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('star.png')
#灰度化
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#二值化
ret,thresh = cv.threshold(img_gray, 127, 255,0)
#检测边缘点
ret, contours,hierarchy = cv.findContours(thresh,2,1)
cnt = contours[0]
#凸包
hull = cv.convexHull(cnt,returnPoints = False)
#凸包中的凹陷
defects = cv.convexityDefects(cnt,hull)
print(defects)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    #画直线
    cv.line(img,start,end,[0,255,0],2)
    #画圆
    cv.circle(img,far,5,[0,0,255],-1)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()

#点多边形测试,这个函数找出图像中一点到轮廓线的最短距离
#第三个参数是measureDist。如果它是真的，它会找到有符号的距离。
#如果为假，则查找该点是在轮廓线内部还是外部(分别返回+1、-1和0)。
dist = cv.pointPolygonTest(cnt,(50,50),False)
print(dist)
