import cv2
img = cv2.imread("frame12.jpg")
n = 0
xwidth = 70;
ywidth = 100;

for col in xrange(0, 2):
    for row in xrange(0, 2):
        crop_img = img[(col*xwidth+0):(col*xwidth+70), (row*ywidth+0):(row*ywidth+100)]
        cv2.imwrite("cropped/frame%d.jpg" % n, crop_img)
        n += 1
        
