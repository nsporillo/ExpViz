import numpy as np
import cv2

img = cv2.imread('simple.jpg',0)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
cv2.imwrite('simple_out.jpg', img)