import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#傅里叶变换
img = cv.imread('new_dog.jpg', 0)
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
#傅里叶变换
dtf_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dtf_shift[:,:,0], dtf_shift[:,:,1]))

rows, cols = img.shape
crow, ccol = rows/2, cols/2
mask = np.zeros((rows, cols, 2), np.float32)
mask[int(crow)-30:int(crow)+30, int(ccol)-30:int(ccol)+30] = 1

fshift = dtf_shift*mask
f_ishift = np.fft.fftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:, 0], img_back[:,:, 1])


plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Output'), plt.xticks([]), plt.yticks([])
plt.show()

'''
#性能优化。当数组大小为2的幂时，速度最快。
#对于大小为2、3和5的乘积的数组，也可以非常有效地进行处理。
rows, clos = img.shape
nrows = cv.getOptimalDFTSize(rows)
ncols = cv.getOptimalDFTSize(cols)
'''