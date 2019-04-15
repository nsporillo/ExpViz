import pickle

import cv2 as cv
import numpy as np
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_model(bin_dir):
    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model


def crop(img):
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    nb_components, _, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(th2), connectivity=8)

    for i in range(1, nb_components):
        # Image stats, sarea is the area of the connected component
        sleft, stop, swidth, sheight, sarea = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], \
                                              stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT], \
                                              stats[i, cv.CC_STAT_AREA]
        # print(sarea, sleft, stop, swidth, sheight)
        # Crop the bounding box of the character
        crp = img[(stop - 2):(stop + sheight + 2), (sleft - 2):(sleft + swidth + 2)]
        return crp, swidth, sheight


def main():
    fonts = [cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_COMPLEX]
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'M', 'S', 'V', 'X', 'Z', 'U',
                'a', 'b', 'c', 'd', 'e', 'i', 'o', 'n', 'y']

    dataset = []

    for a in alphabet:
        for font in fonts:
            img = 255 * np.ones((64, 64), np.uint8)
            img = cv.putText(img, a, (22, 44), font, 1, (0, 0, 0))
            crp, w, h = crop(img)
            dataset.append((crp, a))

    confs = []
    results = []
    x = np.arange(len(dataset))
    bad = []
    for img, a in dataset:
        img = np.invert(cv.resize(img, (28, 28), interpolation=cv.INTER_LANCZOS4))
        ci = img.reshape(1, 28, 28, 1).astype('float32')

        ci /= 255
        result = model.predict(ci)
        conf = str(round(max(result[0]) * 100, 2))
        index = int(np.argmax(result, axis=1)[0])
        confs.append(conf)
        if index < len(mapping):
            value = 1 if a.capitalize() == str(mapping[index]).capitalize() else 0
            results.append(value)
            if value == 0:
                bad.append(img)
            print("\tsymbol=[" + a + "] pred=[" + mapping[index] + "] conf=" + conf)
        else:
            results.append(0)

    confs = np.array(confs, dtype=np.float32) / 100
    print('Accuracy', str(round(np.sum(results) / len(dataset), 3)))
    print('Mean conf', str(round(np.mean(confs), 3)))
    plt.figure(figsize=(12, 6))
    plt.xlabel('Symbol Index')
    plt.ylabel('Confidence')
    plt.plot(x, confs)
    plt.plot(x, results)

    plt.figure(figsize=(14, 8))
    plt.title('Misclassified symbols')
    columns = 5
    for i, image in enumerate(bad):
        plt.subplot(len(bad) / columns + 1, columns, i + 1)
        plt.axis('off')
        plt.imshow(image)

    plt.show()


if __name__ == '__main__':
    # Load trained model and classify the given image
    model = load_model('bin')
    mapping = pickle.load(open('bin/mapping.p', 'rb'))
    main()
