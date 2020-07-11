import cv2 as cv
import numpy as np
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
contours, hierarchy = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
M = cv.moments(cnt)
#print( M )

#轮廓面积
area = cv.contourArea(cnt) 

#轮廓周长
perimeter = cv.arcLength(cnt,True)

#轮廓近视
epsilon = 0.1*cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

#轮廓凸包
hull = cv.convexHull(cnt)
#检查凸度，检测是否有凸出，返回值为True或者False
k = cv.isContourConvex(cnt) 

#边界矩阵
#直角矩阵
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#旋转矩阵
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)

#最小闭合圈
#使用函数**cv.minEnclosingCircle(*()查找对象的圆周
(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv.circle(img,center,radius,(0,255,0),2)

#拟合一个椭圆
ellipse = cv.fitEllipse(cnt)
cv.ellipse(img,ellipse,(0,255,0),2)

#拟合直线
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)



cv.drawContours(img, approx, -1, (0,255,0), 3)
cv.imshow('demo', img)
cv.waitKey()
cv.destroyAllWindows()




'''
cnt = contours[0]
data_x = []
data_y = []
print(cnt[262, 0, 0])
for i in range(263):
    data_x.append(cnt[i, 0, 0])
    data_y.append(cnt[i, 0, 1])
data_x.append(cnt[0, 0, 0])
data_y.append(cnt[0, 0, 1])
data_x = data_x[::-1]
data_y = data_y[::-1]
plt.plot(data_x, data_y)
plt.title('paint')
plt.show()
print(cnt)
M = cv.moments(cnt)
#print( M )
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
#print(cx, cy)
cv.drawContours(img, contours[0], -1, (0, 255, 0), 3)
cv.imshow('img', img)
cv.waitKey()
cv.imwrite('fox.jpg', img)
cv.destroyAllWindows()
'''
