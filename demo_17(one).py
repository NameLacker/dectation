import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Opencv中的轮廓
img = cv.imread('game_image.png')
#print(img.shape)
#转化为灰度图
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#print(img.shape)
#转化为二值图
ret, img2 = cv.threshold(img1, 127, 255, 0)
#print(img.shape)

#检测轮廓并绘制,第一个返回了所处理的图像,第二个是轮廓的点集,第三个，各层轮廓的索引
contours, hierarchy = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
Area = []
#将所有方框的面积检测出来
for m in contours:
    cnt = m
    Area.append(cv.contourArea(cnt))
print(len(Area))
#查找面积最大的下标
sum = 0
ar = 0.0
for i in Area:
    sum = sum + 1
    if i >= ar:
        ar = i
        col = sum
print(col)

cv.drawContours(img, contours[col - 1], -1, (0, 255, 0), 3)
cv.imshow('img', img)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite('test.png', img)

'''
paint = contours[0]
print(paint[60, 0, 1])
paint_x = []
paint_y = []
for i in range(61):
    paint_x.append(paint[i, 0, 0])
    paint_y.append(paint[i, 0, 1])
paint_x.append(paint[0, 0, 0])
paint_y.append(paint[0, 0, 1])
print(len(paint_x))
print(len(paint_y))
plt.plot(paint_x, paint_y)
plt.title('paint')
plt.show()
'''
