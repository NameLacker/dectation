import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

BLUE = [0, 255, 0]
img = cv.imread('cat.jpg')

#添加恒定边框
constant =  cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)
#边缘颜色镜像：fedcba/abcdefgh/hgfedc
reflect = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT)
#较上稍有差异：gfedcb/abcdefgh/gfedcb
reflect_101 = cv.copyMakeBorder(img, 10, 10, 10, 10,cv.BORDER_REFLECT_101)
#同上
default = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_DEFAULT)
#复制边缘颜色：aaaaaa/abcdefgh/hhhhhh
replicate = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REPLICATE)
#cdefgh/abcdefgh/abcdef
wrap = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_WRAP)

plt.subplot(231), plt.imshow(constant, 'gray'), plt.title('constant')
plt.subplot(232), plt.imshow(reflect, 'gray'), plt.title('reflect')
plt.subplot(233), plt.imshow(reflect_101, 'gray'), plt.title('reflect_101')
plt.subplot(234), plt.imshow(default, 'gray'), plt.title('default')
plt.subplot(235), plt.imshow(replicate, 'gray'), plt.title('replicate')
plt.subplot(236), plt.imshow(wrap, 'gray'), plt.title('wrap')
plt.show()