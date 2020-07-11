import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('dog.jpg')
imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img[:, :, 0])
dog_b, dog_g, dog_r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

#图像增强
equ_b = cv.equalizeHist(dog_b)
res_b = np.hstack((dog_b, equ_b))

equ_g = cv.equalizeHist(dog_g)
res_g = np.hstack((dog_g, equ_g))

equ_r = cv.equalizeHist(dog_r)
res_r = np.hstack((dog_r, equ_r))

res = cv.merge((res_b, res_g, res_r))
cv.imshow('img', res)
cv.imwrite('temp.jpg', res)
cv.waitKey()
cv.destroyAllWindows()

# 对比度受限的自适应直方图均衡。这种情况下，图像被分成称为“tiles”的小块（在OpenCV中，tileSize默认为8x8）。
# 然后，像往常一样对这些块中的每一个进行直方图均衡。
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(imggray)
cv.imwrite('clahe_2.jpg',cl1)