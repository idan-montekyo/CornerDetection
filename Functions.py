import numpy as np
import cv2
from matplotlib import pyplot as plt


# Calculates second gradients, filters, and returns them. Prints process.
def secondGradsFiltered(img, display):
    imgx = img.copy()
    imgy = img.copy()

    # 1st order gradients
    ix = cv2.Sobel(imgx, cv2.CV_64F, 1, 0, ksize=5)
    iy = cv2.Sobel(imgy, cv2.CV_64F, 0, 1, ksize=5)

    # 2nd order gradients
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    # Moving-Average mask
    kernel = np.ones((5, 5), np.float32) / 25
    # Filtering 2nd order gradients
    fix2 = cv2.filter2D(ix2, -1, kernel)
    fiy2 = cv2.filter2D(iy2, -1, kernel)
    fixy = cv2.filter2D(ixy, -1, kernel)

    # Printing the gradients:
    if display:
        # plt.figure('1st order gradients')
        plt.figure('Gradients, 2nd order gradients, and filtered 2nd order gradients')
        plt.subplot(331)
        plt.imshow(ix, cmap='gray')
        plt.title('Gx')
        plt.axis('off')
        plt.subplot(332)
        plt.imshow(img, cmap='gray')
        plt.title('Gray-scale')
        plt.axis('off')
        plt.subplot(333)
        plt.imshow(iy, cmap='gray')
        plt.title('Gy')
        plt.axis('off')
        # plt.show()

        # plt.figure('2nd order gradients')
        plt.subplot(334)
        plt.imshow(ix2, cmap='gray')
        plt.title('Gxx')
        plt.axis('off')
        plt.subplot(335)
        plt.imshow(ixy, cmap='gray')
        plt.title('Gxy')
        plt.axis('off')
        plt.subplot(336)
        plt.imshow(iy2, cmap='gray')
        plt.title('Gyy')
        plt.axis('off')
        # plt.show()

        # plt.figure('FILTERED 2nd order gradients')
        plt.subplot(337)
        plt.imshow(fix2, cmap='gray')
        plt.title('Gxx filtered')
        plt.axis('off')
        plt.subplot(338)
        plt.imshow(fixy, cmap='gray')
        plt.title('Gxy filtered')
        plt.axis('off')
        plt.subplot(339)
        plt.imshow(fiy2, cmap='gray')
        plt.title('Gyy filtered')
        plt.axis('off')
        plt.show()

    return fix2, fixy, fiy2


# Calculates and returns score-matrix using Harris equation
def calcRmatViaDetAndTrace(ix2, ixy, iy2):
    rows, cols = ix2.shape
    # Creating score matrix R that holds the probability of each pixel being a corner
    r = np.zeros((rows, cols))
    rmax = 0

    # Moving-Average mask
    kernel = np.ones((5, 5), np.float32) / 25
    # Second filtering to 2nd order gradients
    ix2 = cv2.filter2D(ix2, -1, kernel)
    iy2 = cv2.filter2D(iy2, -1, kernel)
    ixy = cv2.filter2D(ixy, -1, kernel)

    # Calculating the score for each pixel in R
    for i in range(rows):
        for j in range(cols):
            m = np.array([[ix2[i, j], ixy[i, j]],
                          [ixy[i, j], iy2[i, j]]], dtype=np.float64)
            r[i, j] = np.linalg.det(m) - 0.04 * (np.power(np.trace(m), 2))
            if r[i, j] > rmax:
                rmax = r[i, j]

    return r, rmax


# Calculates and returns the result matrix in relative to R-max. Prints process.
def resultMatrix(r, rmax, display):
    rows, cols = r.shape
    # Creating a boolean result matrix that holds the images corners
    result = np.zeros((rows, cols))

    # Changed corner-detection sensitivity from 0.01 to 0.005
    sensitivity = 0.005
    threshold = sensitivity * rmax
    # NMS - Non Maximal Suppression
    # Picking only the pixels which surpass the threshold & are the highest among its neighbors from all directions
    for i in range(rows - 1):
        for j in range(cols - 1):
            if r[i, j] > threshold \
                and r[i, j] > r[i - 1, j - 1] and r[i, j] > r[i - 1, j + 1] \
                and r[i, j] > r[i + 1, j - 1] and r[i, j] > r[i + 1, j + 1] \
                and r[i, j] > r[i, j - 1] and r[i, j] > r[i, j + 1] \
                and r[i, j] > r[i + 1, j] and r[i, j] > r[i - 1, j]:
                result[i, j] = 1

    # Our addition:
    # Pixels that considered to be corners but does not hold the maximum R value in a certain window
    # will no longer be counted as corners.

    # window size = 9
    for i in range(4, rows - 4):
        for j in range(4, cols - 4):
            window = r[i-4: i+5, j-4: j+5]
            if r[i, j] < np.max(window):
                result[i, j] = 0

    # Printing the score and result matrices
    if display:
        plt.figure('Score matrix R and result matrix')
        plt.subplot(121)
        plt.imshow(r, cmap='gray')
        plt.title('score matrix (harris)')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(result, cmap='gray')
        plt.title('result matrix (harris)')
        plt.axis('off')
        plt.show()

    return result


# Printing Before-After of the selected image.
def printResult(img, result):
    if img.ndim == 2:
        img = img.copy()
        img = cv2.merge((img, img, img))

    pc, pr = np.where(result == 1)
    # for x in range(len(pc)):
    #     cv2.circle(img, (pr[x], pc[x]), 5, (255, 0, 0), -1)
    plt.figure('Before and after')
    plt.subplot(121)
    plt.imshow(img[:, :, ::-1])
    plt.title('Before')
    plt.axis('off')
    plt.subplot(122)
    plt.plot(pr, pc, 'b+')
    plt.imshow(img[:, :, ::-1])
    plt.title('After')
    plt.axis('off')
    # plt.savefig('OurAlgo_Results_harris.png', bbox_inches='tight')
    plt.show()


# Checking Input-validity, and executing our version of Harris algorithm.
def myHarris(originalImg, display=False):
    # Validity check
    if not isinstance(display, bool):
        return None
    if not isinstance(originalImg, np.ndarray):
        return None
    if originalImg.ndim != 2 and originalImg.ndim != 3:
        return None
    if originalImg.ndim == 3 and originalImg.shape[2] != 3:
        return None

    if originalImg.ndim == 2:
        imgGray = originalImg.copy()

    else:
        imgGray = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)

    # Algorithm's execution
    ix2, ixy, iy2 = secondGradsFiltered(imgGray, display)
    r, rmax = calcRmatViaDetAndTrace(ix2, ixy, iy2)
    result = resultMatrix(r, rmax, display)
    printResult(originalImg, result)
