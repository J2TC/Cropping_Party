import cv2

for n in xrange(1,450):
  original = cv2.imread('frames/snaps/original (%d).png' % n)
  crop_img = original[300: 400, 65:1215]
  cv2.imwrite("frames/frame%d.jpg" % n, crop_img)
