import cv2
from tqdm import tqdm


for n in tqdm(xrange(1,450)):
  original = cv2.imread('frames/snaps/original (%d).png' % n)
  crop_img = original[350: 420, 400:1100]
  cv2.imwrite("frames/frame%d.jpg" % n, crop_img)
