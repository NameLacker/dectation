import numpy as np
import cv2 as cv

'''
#改变颜色空间
img = cv.imread('dog.jpg')
img_1 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

green = np.uint8([[[255, 0, 0]]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print(hsv_green)

cv.imshow('green', green)
cv.waitKey()
print(hsv_green)
cv.destroyAllWindows()
'''

#图像的几何变换

'''
#缩放
img = cv.imread('dog.jpg')
#缩小一半
res = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
#放大一倍
heigh, weight = img.shape[:2]
res_1 = cv.resize(img, (2*weight, 2*heigh), interpolation=cv.INTER_CUBIC)
print(img.shape)
print(res.shape)
print(res_1.shape)
'''

'''
#平移
img = cv.imread('dog.jpg')
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 100]])
dst = cv.warpAffine(img, M, (col    s, rows))
'''

'''
#旋转
img = cv.imread('logo.jpg')
rows, cols = img.shape[:2]
M = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1)
dst = cv.warpAffine(img, M, (cols, rows))
'''

'''
#仿射变换
img = cv.imread('cat.jpg')
rows, cols, ch = img.shape[:]
pst1 = np.float32([[20, 0], [20, 20], [30, 20]])
pst2 = np.float32([[30, 0], [20, 20], [30, 20]])
M = cv.getAffineTransform(pst1, pst2)
dst = cv.warpAffine(img, M, (cols, rows))
'''

#透视变换
img = cv.imread('dog.jpg')
rows, cols, ch = img.shape
print(img.shape)
pst1 = np.float32([[220, 530], [1445, 23], [701, 1731], [1101, 1801]])
pst2 = np.float32([[0, 0], [900, 0], [0, 1800], [900, 1800]])
M = cv.getPerspectiveTransform(pst1, pst2)
dst = cv.warpPerspective(img, M, (cols, rows))


cv.imshow('res_1', dst)
cv.imwrite('dog_dst_sin.jpg', dst)
cv.waitKey()
cv.destroyAllWindows()
