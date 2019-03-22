import os
import sys
import glob

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

imgSize = (1288, 966)


def load(path):
    """
    Loads the input image from file
    :param path: The relative path to the input image file
    :return: The resized input image in grayscale with red channel removed beforehand
    """
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    if image is not None:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        print('Error: (File not found): ' + str(path))
        sys.exit(0)


def to_bgr(im):
    """
    Converts a grayscale image back to BGR to allow for color
    :param im: Grayscale image
    :return: Same image but with 3 channels for BGR
    """
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (23, 30), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    return img


def process(img):
    blur = cv.medianBlur(cv.medianBlur(img, 3), 3)
    _, th2 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th2 = cv.dilate(th2, np.ones((5, 5), dtype=np.uint8), iterations=2)
    th2 = cv.erode(th2, np.ones((5, 5), dtype=np.uint8), iterations=1)

    nb_components, _, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(th2), connectivity=8)

    for i in range(1, nb_components):
        # Image stats, sarea is the area of the connected component
        sleft, stop, swidth, sheight, sarea = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], \
                                              stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT], \
                                              stats[i, cv.CC_STAT_AREA]

        if 8 < swidth < 128 and 8 < sheight < 128:
            #print(sarea, sleft, stop, swidth, sheight)
            # Crop the bounding box of the character
            crp = img[stop:(stop + sheight), sleft:(sleft + swidth)]


def main():
    if len(sys.argv) > 0:
        path = str(sys.argv[1])
        print('INPUT Filename: ' + path)
        process(load(path))
        print('finished')
    else:
        print('Usage: preprocess.py <filename>')


main()
