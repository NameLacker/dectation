
'''
opencv读写图片
读出摄像头
'''


import numpy as np
import cv2 as cv

'''
img = cv.imread('cat.jpg', 0)
cv.imshow('cat', img)
Key = cv.waitKey()
if Key == ord('s'):
    cv.imwrite('cat.jpeg', img)
'''

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open the Video')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame!")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()