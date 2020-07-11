import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(1):
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #定义蓝色色彩范围
    low_white = np.array([100, 50, 50])
    top_white = np.array([130, 255, 255])

    mask = cv.inRange(hsv, low_white, top_white)
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()