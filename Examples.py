# --------------------------------------- Ready piece of code - Shi-Tomasi ----------------------------------------
#
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
#
# img = cv2.imread('Pictures/harris.png')
# img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
# corners = np.int0(corners)
#
# for corner in corners:
# 	x, y = corner.ravel()
# 	cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
#
# cv2.imshow('Frame', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------
# img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# --------------------------------------- Original Harris Algo -----------------------------------------------

# # Comparison to harris:
# imgHarris = img1.copy()
# gray = np.float32(imgGray)
# harrisImg = cv2.cornerHarris(gray, 5, 5, 0.04)
# harrisImg = cv2.dilate(harrisImg,None)
# imgHarris[harrisImg>0.01*harrisImg.max()]=[0,0,255]
# plt.figure('Harris Corner Detector')
# plt.subplot(121)
# plt.imshow(img2[:, :, ::-1])
# plt.title('Original Image')
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(imgHarris[:, :, ::-1])
# plt.title('Original Harris Algo')
# plt.axis('off')
# # plt.savefig('OriginalHarrisAlgo_harris.png', bbox_inches='tight')
# plt.show()
