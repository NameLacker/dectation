'''
opencv画图
'''



import numpy as np
import cv2 as cv

img = np.ones((512, 512, 3), np.uint8)


#画线
#print(img)
img_1 = cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
cv.imshow('paint', img_1)
#cv.waitKey()

#画矩形
img_2 = cv.rectangle(img, (110, 110), (510, 510), (255, 0, 0), 5)
cv.imshow('paint',img_2)
#cv.waitKey()

#画圆
img_3 = cv.circle(img, (255, 255), 150, (255, 0, 0), -1)
cv.imshow('paint',img_3)
#cv.waitKey()

#画椭圆
img_4 = cv.ellipse(img, (256,256), (100,50), 90, 0,360, (255, 255, 0), -1)
cv.imshow('paint', img_4)
#cv.waitKey()

#画多边形
pts = np.array([[5,5],[507,5],[507,507],[5,507]], np.int32)
pts = pts.reshape((-1,1,2))
#print(pts)
img_5 = cv.polylines(img,[pts],True,(0,255,255), 3)
cv.imshow('paint', img_5)
cv.waitKey()

cv.destroyAllWindows()


