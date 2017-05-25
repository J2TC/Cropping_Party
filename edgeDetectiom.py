import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread('frames/frame125.jpg',0)
img_original = img
img_gauss = cv2.blur(img,(10,10))

sobelx = cv2.Sobel(img_original,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img_original,cv2.CV_64F,0,1,ksize=5)
sobel2 = sobelx+sobely

sobelx_blur = cv2.Sobel(img_gauss,cv2.CV_64F,1,0,ksize=5)
sobely_blur = cv2.Sobel(img_gauss,cv2.CV_64F,0,1,ksize=5)
sobel2_blur = sobelx+sobely

canny = cv2.Canny(img_original,100,200)
canny_blur = cv2.Canny(img_gauss,100,200)

plt.subplot(3,2,1),plt.imshow(img_original,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(img,cmap = 'gray')
plt.title('GaussianBlur'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,3),plt.imshow(sobel2,cmap = 'gray')
plt.title('Sobel 2 Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,4),plt.imshow(sobel2_blur,cmap = 'gray')
plt.title('Sobel2 Gaussian'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,6),plt.imshow(canny_blur,cmap = 'gray')
plt.title('Canny Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,5),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])


plt.show()
