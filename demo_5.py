'''
轨迹栏作为调色板
'''


import cv2 as cv
import numpy as np

def nothing(x):
    pass

img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('image')
cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)
switch = '1 ON \n0 OFF'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while True:
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('g', 'image')
    b = cv.getTrackbarPos('b', 'image')
    s = cv.getTrackbarPos(switch, 'image')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]
cv.destroyAllWindows()