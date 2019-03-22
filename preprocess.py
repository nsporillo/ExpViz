import sys
import cv2 as cv
import numpy as np
try:
    import cv2.__init__ as cv
except ImportError:
    pass

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


def process(img):
    blur = cv.medianBlur(cv.medianBlur(img, 3), 3)
    _, th2 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th2 = cv.dilate(th2, np.ones((5, 5), dtype=np.uint8), iterations=2)
    th2 = cv.erode(th2, np.ones((5, 5), dtype=np.uint8), iterations=1)
    th3 = cv.morphologyEx(th2, cv.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
    nb_components, _, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(th3), connectivity=8)

    bounds = to_bgr(th2)

    new = []
    for i in range(1, nb_components):
        over = False
        sleft, stop, swidth, sheight, sarea = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], \
                                              stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT], \
                                              stats[i, cv.CC_STAT_AREA]
        for j in range(1, nb_components):
            jsleft, jstop, jswidth, jsheight, jsarea = stats[j, cv.CC_STAT_LEFT], stats[j, cv.CC_STAT_TOP], \
                                                  stats[j, cv.CC_STAT_WIDTH], stats[j, cv.CC_STAT_HEIGHT], \
                                                  stats[j, cv.CC_STAT_AREA]
            if sleft<jsleft<sleft+swidth or jsleft<sleft<jsleft+jswidth:
                over = True
                bound = (min(sleft,jsleft),min(stop,jstop),max(jswidth-sleft+jsleft,swidth-jsleft+sleft),
                         max(jsheight+stop-jstop,sheight+jstop-stop))
                #cv.rectangle(bounds, (bound[0], bound[1]), (bound[0]+bound[2], bound[1]+bound[3]), (0, 0, 255), thickness=2)
                over = True
                new.append(bound)
        if not over:
            new.append((sleft, stop, swidth, sheight))

    """for i in range(1, nb_components):
        # Image stats, sarea is the area of the connected component
        sleft, stop, swidth, sheight, sarea = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], \
                                              stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT], \
                                              stats[i, cv.CC_STAT_AREA]

        if 8 < swidth and 8 < sheight:
            #print(sarea, sleft, stop, swidth, sheight)
            # Crop the bounding box of the character
            crp = img[stop:(stop + sheight), sleft:(sleft + swidth)]
            cv.rectangle(bounds, (sleft, stop), (sleft + swidth, stop + sheight), (0, 0, 255), thickness=2)
    """
    for i in new:
        cv.rectangle(bounds, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 0, 255), thickness=2)

    cv.imshow('g', bounds)
    cv.waitKey()

def main():
    if len(sys.argv) > 0:
        path = str(sys.argv[1])
        print('INPUT Filename: ' + path)
        process(load(path))
        print('finished')
    else:
        print('Usage: preprocess.py <filename>')


main()
