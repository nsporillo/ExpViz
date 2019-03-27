import argparse
import os
import pickle
import cv2 as cv

import hasy_tools
import keras
import numpy as np
from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat


def load_emnist_data(mat_file_path, width=28, height=28):
    mat = loadmat(mat_file_path)

    # Load character mapping
    mapping = {kv[0]: chr(kv[1:][0]) for kv in mat['dataset'][0][0][2]}

    # Load training data
    max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]
    print(training_labels[0], training_labels[1])

    # Load testing data
    max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data
    _len = len(training_images)
    for i in range(len(training_images)):
        training_images[i] = np.rot90(np.fliplr(training_images[i]))

    # Reshape testing data
    _len = len(testing_images)
    for i in range(len(testing_images)):
        testing_images[i] = np.rot90(np.fliplr(testing_images[i]))

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')


    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return (training_images, training_labels), (testing_images, testing_labels), nb_classes, mapping


def load_hasy_data(width=28, height=28):
    data = hasy_tools.load_data()
    # Load training data
    max_ = data['x_train'].shape[0]
    pre_train_images = (data['x_train'][:max_]).reshape(max_, 32, 32, 1)
    training_images = np.zeros((max_, 28, 28, 1), dtype=np.float32)
    for i in range(0, max_):
        training_images[i] = np.expand_dims(
            np.invert(cv.resize(pre_train_images[i], (height, width), interpolation=cv.INTER_LANCZOS4)), axis=2)
    del pre_train_images
    training_labels = data['y_train'][:max_] + 62
    print('Hasy', training_labels[0], training_labels[max_-2])

    max_ = data['x_test'].shape[0]
    pre_test_images = (data['x_test'][:max_]).reshape(max_, 32, 32, 1)
    testing_images = np.zeros((max_, 28, 28, 1), dtype=np.float32)
    for i in range(0, max_):
        testing_images[i] = np.expand_dims(
            np.invert(cv.resize(pre_test_images[i], (height, width), interpolation=cv.INTER_LANCZOS4)), axis=2)
    del pre_test_images
    testing_labels = data['y_test'][:max_] + 62

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    return (training_images, training_labels), (testing_images, testing_labels), len(data['labels']), data['labels']


def build_net(num_classes, width=28, height=28):
    input_shape = (height, width, 1)

    model = Sequential()
    model.add(Convolution2D(32,
                            (3, 3),
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(64,
                            (3, 3),
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print(model.summary())
    return model


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


def train(model, training_data, num_classes, batch_size=256, epochs=5):
    x_train, y_train, x_test, y_test = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

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
    hasy_data = load_hasy_data()
    hasy_mapping = hasy_data[3]
    print('Hasy data loaded ' + str(hasy_data[2]) + ' classes')

    mapping = emnist_mapping
    k = len(mapping)
    for i in range(0, len(hasy_mapping)):
        mapping[k+i] = hasy_mapping[i]

    print(mapping)
    num_classes = emnist_data[2] + hasy_data[2]
    pickle.dump(mapping, open('bin/mapping.p', 'wb'))
    model = build_net(num_classes)
    train(model, merge(emnist_data, hasy_data), num_classes, epochs=args.epochs)
