import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Test.png',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
canny = cv2.Canny(img,100,200)
cannySobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
cannySobel = cv2.Canny((cannySobel ,cmap = 'gray'),100,200)

plt.subplot(2,1,1),plt.imshow(canny&Sobel,cmap = 'gray')
plt.title('canny&Sobel'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,2,3),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])



plt.show()


#Improved sobel

#sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#abs_sobel64f = np.absolute(sobelx64f)