# ------------------------------------------ NOT IN USE -----------------------------------------

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL.Image import *
from scipy import ndimage


def corner(img):
    # X-axis
    img1 = img.copy()
    ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    plt.figure('SobelX')
    plt.imshow(ix, cmap='gray')

    # Y-axis
    img2 = img.copy()
    iy = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)
    plt.figure('SobelY')
    plt.imshow(iy, cmap='gray')

    ixsq = cv2.multiply(ix, ix)
    ix2 = cv2.GaussianBlur(ixsq, (3, 3), 0)
    iysq = cv2.multiply(iy, iy)
    iy2 = cv2.GaussianBlur(iysq, (3, 3), 0)
    ixiy = cv2.multiply(ix, iy)
    ixy = cv2.GaussianBlur(ixiy, (3, 3), 0)

    # Determinant
    a = cv2.multiply(ix2, iy2)
    b = cv2.multiply(ixy, ixy)
    det = cv2.subtract(a, b)

    # Trace
    trc = cv2.add(ix2, iy2)
    trc2 = 0.04 * cv2.multiply(trc, trc)    # k = 0.04

    # Corner score
    cnr = cv2.subtract(det, trc2)   # cnr = R (map)
    plt.figure('Corner score')
    plt.imshow(cnr, cmap='gray')

    # threshold -> each value over 120 will be set to 255
    newCnr = cv2.blur(cnr, (3, 3))
    ret, thresh = cv2.threshold(newCnr, 120, 255, cv2.THRESH_BINARY)   # ret = 2nd slot
    plt.figure('Thresh')
    plt.imshow(thresh, cmap='gray')
    print(ret)
    plt.show()
    return thresh

