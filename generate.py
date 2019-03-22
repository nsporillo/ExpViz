import os
import sys
import glob

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

datadir = 'data/'


def crop(img):
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th2 = cv.erode(th2, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=2)
    th2 = cv.dilate(th2, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=2)
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
            return crp, swidth, sheight


def main():
    # Create data directory if missing
    if not os.path.exists(datadir):
        os.mkdir('data/')
        os.mkdir('data/images/')
    # Clear the old image dataset
    files = glob.glob(datadir + 'images/*')
    for f in files:
        os.remove(f)

    fonts = [cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_COMPLEX]
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                'l', 'o', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '=', '*', '+', '-', '(', ')']
    blurs = [1, 3, 5]

    x = 0
    labels = open("data/training_lbls.txt", "w")
    dataset = []
    stats = []

    for font in fonts:
        for a in alphabet:
            for blur in blurs:
                img = 255 * np.ones((128, 128), np.uint8)
                img = cv.putText(img, a, (44, 88), font, 1, (0, 0, 0), 3)
                img = cv.GaussianBlur(img, (blur, blur), 0)
                crp, w, h = crop(img)
                dataset.append((crp, a))
                stats.append((w, h))

    med = np.median(np.array(stats).astype((np.uint8, np.uint8)), axis=0)

    for img, a in dataset:
        ifn = "data/images/" + str(x) + ".png"
        img = cv.resize(img, (int(med[0]), int(med[1])), interpolation=cv.INTER_LANCZOS4)
        cv.imwrite(ifn, img)
        labels.write(ifn + ' ' + a + '\n')
        x += 1

    print('Median Width Height: ', str(med))

    labels.close()


main()
