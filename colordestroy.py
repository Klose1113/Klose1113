import cv2
import numpy as np
import matplotlib.pylab as plt

low_green = np.array([25, 52, 70])
high_green = np.array([102, 255, 255])

low_destroy = np.array([18,0,140])
high_destroy = np.array([179,255,255])


imag = cv2.imread('road9.jpg')
img = cv2.resize(imag, (900, 650), interpolation=cv2.INTER_CUBIC)
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(imgHSV, low_destroy, high_destroy)

#mask = 255-mask

mask_2 = cv2.inRange(imgHSV, low_green, high_green)

mask_2 = 255-mask_2

res = cv2.bitwise_and(img, img, mask = mask)

res_2 = cv2.bitwise_and(res, res, mask = mask_2)

#cv2.imshow("mask", mask)
cv2.imshow("cam", img)
cv2.imshow('res',res)
#cv2.imshow('res_2', res_2)
cv2.waitKey()