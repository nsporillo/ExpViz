import os
import sys
import glob

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Clear the old image dataset
    files = glob.glob('/images/*')
    for f in files:
        os.remove(f)

    fonts = [cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_COMPLEX]
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                'l', 'o', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '=', '*']
    blurs = [1, 3, 5]

    x = 0
    labels = open("labels/training.txt", "w")
    for font in fonts:
        for a in alphabet:
            for blur in blurs:
                img = 255 * np.ones((64, 64), np.float32)
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                img = cv.putText(img, a, (22, 44), font, 1, (0, 0, 0), 3)
                img = cv.GaussianBlur(img, (blur, blur), 0)
                ifn = 'images/' + str(x) + '.png'
                cv.imwrite(ifn, img)
                labels.write(ifn + ' ' + a + '\n')
                x += 1

    labels.close()

main()
