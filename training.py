import argparse
import os
import pickle
import time

import cv2 as cv
import hasy_tools
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from scipy.io import loadmat


def load_emnist_data(mat_file_path):
    mat = loadmat(mat_file_path)

    # Load character mapping
    mapping = {kv[0]: chr(kv[1:][0]) for kv in mat['dataset'][0][0][2]}

    # Load training data
    max_ = len(mat['dataset'][0][0][0][0][0][0])
    train_imgs = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, 28, 28, 1)
    train_lbls = mat['dataset'][0][0][0][0][0][1][:max_]

    # Reshape training data
    _len = len(train_imgs)
    for trimg in range(_len):
        train_imgs[trimg] = np.rot90(np.fliplr(train_imgs[trimg]))

    # Load testing data
    max_ = int(max_ / 6)
    test_imgs = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, 28, 28, 1)
    test_lbls = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape testing data
    _len = len(test_imgs)
    for teimg in range(_len):
        test_imgs[teimg] = np.rot90(np.fliplr(test_imgs[teimg]))

    train_imgs = train_imgs.astype('float32')
    test_imgs = test_imgs.astype('float32')

    # Normalize to prevent issues with model
    train_imgs /= 255
    test_imgs /= 255

    return (train_imgs, train_lbls), (test_imgs, test_lbls), len(mapping), mapping


def load_hasy_data(start=0, classes=None):
    data = hasy_tools.load_data()
    print("Hasy Labels: " + str(data['labels']))
    indices = []
    realindices = {}
    i = 0
    if classes is not None:
        for clazz in classes:
            index = data['labels'].index(clazz)
            indices.append(index)
            realindices[index] = int(start + i)
            i += 1

    print("Chosen Hasy Class Indices: " + str(indices) + ', Mapping = [' + str(realindices.values) + ']')

    # Load training data
    training_labels = data['y_train'][:data['x_train'].shape[0]]
    pre_train_images, pre_train_labels = [], []

    for i in range(0, len(training_labels)):
        label = training_labels[i]
        if indices.__contains__(label):
            pre_train_images.append(data['x_train'][i])
            pre_train_labels.append(float(realindices[int(label)]))

    max_ = len(pre_train_images)

    print("Loading " + str(len(pre_train_images)) + " Hasy training images")
    pre_train_images = np.array(pre_train_images).reshape(len(pre_train_images), 32, 32, 1)
    training_images = np.zeros((max_, 28, 28, 1))
    for i in range(0, max_):
        training_images[i] = np.expand_dims(cv.resize(np.invert(pre_train_images[i]), (28, 28)), axis=2)
    del pre_train_images

    training_labels = np.array(pre_train_labels)

    # Load Testing data
    testing_labels = data['y_test'][:data['x_test'].shape[0]]
    pre_test_images, pre_test_labels = [], []

    for i in range(0, len(testing_labels)):
        label = testing_labels[i]
        if indices.__contains__(label):
            pre_test_images.append(data['x_test'][i])
            pre_test_labels.append(float(realindices[int(label)]))

    max_ = len(pre_test_images)

    print("Loading " + str(len(pre_test_images)) + " Hasy testing images")
    pre_test_images = np.array(pre_test_images).reshape(len(pre_test_images), 32, 32, 1)
    testing_images = np.zeros((max_, 28, 28, 1))
    for i in range(0, max_):
        testing_images[i] = np.expand_dims(cv.resize(np.invert(pre_test_images[i]), (28, 28)), axis=2)
    del pre_test_images

    testing_labels = np.array(pre_test_labels)

    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    if classes is not None:
        return (training_images, training_labels), (testing_images, testing_labels), len(classes), classes
    else:
        return (training_images, training_labels), (testing_images, testing_labels), len(data['labels']), data['labels']


def merge(emnist, hasy):
    (x1_train, y1_train), (x1_test, y1_test), _, _ = emnist
    (x2_train, y2_train), (x2_test, y2_test), _, _ = hasy

    print(y1_train.shape, y1_test.shape, y2_train.shape, y2_test.shape)
    x_train = np.zeros((x1_train.shape[0] + x2_train.shape[0], 28, 28, 1))

    for i in range(0, x1_train.shape[0]):
        x_train[i] = x1_train[i]
    for i in range(0, x2_train.shape[0]):
        x_train[x1_train.shape[0] + i] = x2_train[i]

    y_train = np.zeros((y1_train.shape[0] + y2_train.shape[0], 1))
    for i in range(0, y1_train.shape[0]):
        y_train[i] = y1_train[i]
    for i in range(0, y2_train.shape[0]):
        y_train[y1_train.shape[0] + i] = y2_train[i]

    x_test = np.zeros((x1_test.shape[0] + x2_test.shape[0], 28, 28, 1))

    for i in range(0, x1_test.shape[0]):
        x_test[i] = x1_test[i]
    for i in range(0, x2_test.shape[0]):
        x_test[x1_test.shape[0] + i] = x2_test[i]

    y_test = np.zeros((y1_test.shape[0] + y2_test.shape[0], 1))
    for i in range(0, y1_test.shape[0]):
        y_test[i] = y1_test[i]
    for i in range(0, y2_test.shape[0]):
        y_test[y1_test.shape[0] + i] = y2_test[i]

    return x_train, y_train, x_test, y_test


def build_net(num_classes, width=28, height=28):
    input_shape = (height, width, 1)

    model = Sequential()
    model.add(Convolution2D(128,
                            (5, 5),
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(32,
                            (3, 3),
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def train(model, training_data, mapping, num_classes, batch_size=256, epochs=10, plot=True):
    x_train, y_train, x_test, y_test = training_data

    start = time.time()

    if plot:
        # Plot training data symbol histogram
        labels = list(mapping.values())
        ind = np.arange(num_classes)
        width = .1
        plt.figure()
        hist = np.histogram(y_train, bins=num_classes)[0]
        plt.bar(ind + width, hist, align='center', tick_label=labels)
        plt.gca().set(title='Character Frequency Histogram', ylabel='Occurrences')

    print(str(np.unique(y_train)))
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.3,
        fill_mode='nearest')

    datagen.fit(x_train)

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(x_test, y_test))

    end = time.time()
    print('Training took ' + str(end - start) + ' seconds')

    if plot:
        # Plot training & validation accuracy values
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Save the model to file
    model_yaml = model.to_yaml()
    with open("bin/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Program to train a CNN for character detection')
    parser.add_argument('-f', '--file', type=str, help='File to use for training', required=True)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    emnist_data = load_emnist_data(args.file)
    emnist_mapping = emnist_data[3]
    print('EMNIST data loaded ' + str(emnist_data[2]) + ' classes')
    hasy_data = load_hasy_data(len(emnist_mapping),
                               ['+', '-', '\\$', '\\pi', '\\{', '\\}', '\\forall', '\\doteq', '\\pm', '\\nabla'])
    hasy_mapping = hasy_data[3]
    print('Hasy data loaded ' + str(hasy_data[2]) + ' classes')

    mapping = emnist_mapping
    print(emnist_mapping)
    print(hasy_mapping)
    k = len(mapping)
    for i in range(0, len(hasy_mapping)):
        mapping[k + i] = hasy_mapping[i]

    print(mapping)
    num_classes = len(mapping)
    print('Total classes: ' + str(num_classes))
    pickle.dump(mapping, open('bin/mapping.p', 'wb'))
    model = build_net(num_classes)
    train(model, merge(emnist_data, hasy_data), mapping, num_classes, epochs=args.epochs)
