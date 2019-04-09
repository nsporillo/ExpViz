import cv2 as cv
import scipy.spatial.kdtree as kdtree
import numpy as np


def flood_fill(img):
    im_f = cv.bitwise_not(img).copy()
    # Pad all edges of the image with zeros. Super important step
    im_f = np.pad(im_f, ((10, 10), (10, 10)), 'constant')
    h, w = im_f.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(im_f, mask, (0, 0), 0)
    return cv.bitwise_not(im_f)


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
    coords = np.column_stack(np.where(img < 255))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)


def padto(shape, img):
    height, width = img.shape
    dheight, dwidth = shape
    xpad = dwidth - width
    xtuple = (int(xpad / 2), xpad - int(xpad / 2))
    ypad = dheight - height
    ytuple = (int(ypad / 2), ypad - int(ypad / 2))
    return np.pad(img, (ytuple, xtuple), 'constant', constant_values=(255))


def radialJoin(img, stats, centroids):
    stats = np.append(stats, centroids, axis=1)
    combine = np.array(sorted(stats, key=lambda x: (x[0], x[1])))
    centroids = list(combine[:, -2:])
    stats = list(combine[:, :-2])
    centroids = centroids[1:]
    stats = stats[1:]
    centroidTree = kdtree.KDTree(centroids, 10)
    nearests = centroidTree.query(centroids, 5)[1][:, 1:]
    covered = {}
    results = []
    for x in range(len(nearests)):
        over = False
        sleft, stop, swidth, sheight, sarea = stats[x][0], stats[x][1], stats[x][2], stats[x][3], stats[x][4]
        if sarea < 100:
            continue
        for y in range(4):
            if over:
                continue
            if x in covered or nearests[x][y] in covered:
                continue
            n = centroids[nearests[x][y]]
            c = centroids[x]

            nsleft, nstop, nswidth, nsheight, nsarea = stats[nearests[x][y]][0], stats[nearests[x][y]][1], \
                                                       stats[nearests[x][y]][2], stats[nearests[x][y]][3], \
                                                       stats[nearests[x][y]][4]
            if nsarea < 100 or abs(nswidth - swidth) > 100:
                continue
            if sleft - 10 <= nsleft <= sleft + swidth + 10 or nsleft - 10 <= sleft <= nsleft + nswidth + 10:
                bound = (
                min(sleft, nsleft), min(stop, nstop), max(sleft + swidth, nsleft + nswidth) - min(sleft, nsleft),
                max(stop + sheight, nstop + nsheight) - min(stop, nstop), sarea + nsarea)
                # cv.rectangle(bounds, (bound[0], bound[1]), (bound[0]+bound[2], bound[1]+bound[3]), (0, 0, 255), thickness=2)
                results.append(np.array(bound, dtype=np.uint32))
                covered[nearests[x][y]] = ""
                print("connecting: " + str(n) + "," + str(c))
                over = True
        if not over and not x in covered:
            results.append(np.array(stats[x], dtype=np.uint32))
        covered[x] = ""
    return results
