import sys
import argparse
from keras.models import model_from_yaml
from matplotlib import pyplot as plt
import pickle

from preprocessing import *

model = None
mapping = None

lookup = {'\\doteq':'=', '\\neg':'-', '\\ast':'*',
                                '[':'(', ']':')'}


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


def plot_images(imgs, title):
    fig = plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.axis('off')
    columns, rows = 4, 5
    ax = []
    for i in range(0, min(len(imgs), 20)):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        img = cv.applyColorMap(imgs[i], cv.COLORMAP_BONE)
        plt.imshow(img)
    plt.show()


def process(img, minconf, debug=False):
    blur = cv.medianBlur(cv.medianBlur(img, 3), 3)
    _, th2 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #th2 = cv.dilate(th2, np.ones((3, 3), dtype=np.uint8), iterations=1)
    th2 = cv.erode(th2, np.ones((5, 5), dtype=np.uint8), iterations=2)
    #th2 = cv.dilate(th2, np.ones((5, 5), dtype=np.uint8), iterations=1)
    th2 = deskew(th2)
    nb_components, _, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(th2), connectivity=8)
    stats = radialJoin(img, stats, centroids)
    stats = sorted(stats, key=lambda x: (x[0], x[1]))
    bounds = to_bgr(th2)
    res = to_bgr(np.ones((th2.shape[0], th2.shape[1]), dtype=np.uint8) * 255)
    chars = []
    areas = 0
    equation = ""
    for i in range(len(stats)):
        sleft, stop, swidth, sheight, sarea = stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]
        if sarea > 500:
            areas += sarea
            # Crop the bounding box of the symbol y:y+h, x:x+w
            dbbox = th2[(stop):(stop + sheight), (sleft):(sleft + swidth)]
            scale = min(28.0 / swidth, 28.0 / sheight)
            ci = np.invert(padto((28, 28), cv.resize(flood_fill(dbbox),
                                                     (int(round(swidth * scale)), int(round(sheight * scale))))))
            # ci = np.invert(cv.resize(flood_fill(dbbox), (28, 28)))
            chars.append(ci)
            ci = ci.reshape(1, 28, 28, 1).astype('float32') / 255
            result = model.predict(ci)
            conf = str(round(max(result[0]) * 100, 2))
            # TODO: over prioritize math symbols
            if float(conf) > float(minconf):
                index = int(np.argmax(result, axis=1)[0])
                print("Pred= [" + mapping[index] + "] conf= " + conf + " area=" + str(sarea))
                c = mapping[index]
                if mapping[index] in lookup:
                    c = lookup[mapping[index]]
                equation += c
                if debug:
                    cv.putText(res, mapping[index], (sleft, stop), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
            if debug:
                cv.rectangle(bounds, (sleft, stop), (sleft + swidth, stop + sheight), (0, 0, 255), thickness=2)

    if debug:
        print('Average area: ' + str(areas / (nb_components - 1)))

        bounds = cv.resize(bounds, None, fx=0.5, fy=0.5)
        res = cv.resize(res, None, fx=0.5, fy=0.5)
        plt.axis("off")
        plt.imshow(np.vstack((bounds, res)))
        plot_images(chars, 'Chars')
    return equation

def primer():
    global model,mapping
    model = load_model('bin')
    mapping = pickle.load(open('bin/mapping.p', 'rb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify symbols in an image.')
    parser.add_argument('-f', '--file', type=str, help='Image file', required=True)
    parser.add_argument('-mc', '--minconf', type=str, default='5.0', help='Minimum Confidence Threshold')
    args = parser.parse_args()

    # Load trained model and classify the given image
    model = load_model('bin')
    mapping = pickle.load(open('bin/mapping.p', 'rb'))
    process(load(args.file), minconf=args.minconf)
