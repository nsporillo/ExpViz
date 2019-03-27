import os
import sys
import cv2 as cv
import argparse
from keras.models import model_from_yaml
import numpy as np
import pickle


def load_model(bin_dir):
    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model


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


def process(img, minconf):
    blur = cv.medianBlur(cv.medianBlur(img, 3), 3)
    _, th2 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    th2 = cv.dilate(th2, np.ones((5, 5), dtype=np.uint8), iterations=2)
    th2 = cv.erode(th2, np.ones((3, 3), dtype=np.uint8), iterations=1)
    nb_components, _, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(th2), connectivity=8)
    stats = sorted(stats, key=lambda x: (x[0], x[1]))
    bounds = to_bgr(th2)
    symbols = []

    for i in range(1, nb_components):
        sleft, stop, swidth, sheight, sarea = stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]
        dbbox = th2[stop:(stop + sheight), sleft:(sleft + swidth)]
        ci = flood_fill(dbbox)
        #cv.imshow('Connected Component', ci)
        #cv.waitKey()
        # Crop the bounding box of the symbol y:y+h, x:x+w
        symbols.append(ci)
        cv.rectangle(bounds, (sleft, stop), (sleft + swidth, stop + sheight), (0, 0, 255), thickness=2)

    print(len(symbols), len(mapping))
    for s in symbols:
        s = cv.resize(s, (28, 28))
        sym = np.invert(s).reshape(1, 28, 28, 1).astype('float32')
        sym /= 255
        result = model.predict(sym)
        conf = str(max(result[0]) * 100)
        if float(conf) > float(minconf):
            index = int(np.argmax(result, axis=1)[0])
            print("Prediction: [" + mapping[index] + "] Confidence: " + conf)

    cv.imshow('Boxes', bounds)
    cv.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify symbols in an image.')
    parser.add_argument('-f', '--file', type=str, help='Image file', required=True)
    parser.add_argument('-mc', '--minconf', type=str, default='40.0', help='Minimum Confidence Threshold')
    args = parser.parse_args()

    # Load trained model and classify the given image
    model = load_model('bin')
    mapping = pickle.load(open('bin/mapping.p', 'rb'))
    process(load(args.file), minconf=args.minconf)
