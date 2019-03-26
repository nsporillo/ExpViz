import argparse
import os
import pickle

import keras
import hasy_tools
import numpy as np
import tensorflow as tf
from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat


def load_emnist_data(mat_file_path, width=28, height=28):
    mat = loadmat(mat_file_path)

    # Load character mapping
    mapping = {kv[0]: kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

    # Load training data
    max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

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

    return (training_images, training_labels), (testing_images, testing_labels), nb_classes


def load_hasy_data(height=28, width=28):
    data = hasy_tools.load_data()
    # Load training data
    max_ = data['x_train'].shape[0]
    training_images = (data['x_train'][:max_]).reshape(max_, 32, 32, 1)
    training_images = tf.cast(training_images, tf.float32)
    training_images = tf.image.resize_images(training_images, (height, width))
    training_labels = data['y_train'][:max_]

    max_ = data['x_test'].shape[0]
    testing_images = (data['x_test'][:max_]).reshape(max_, 32, 32, 1)
    testing_images = tf.cast(testing_images, tf.float32)
    testing_images = tf.image.resize_images(testing_images, (height, width))
    testing_labels = data['y_test'][:max_]

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    return (training_images, training_labels), (testing_images, testing_labels), len(data['labels'])


def combine_data(src1, src2):
    (x1_train, y1_train), (x1_test, y1_test), c1 = src1
    (x2_train, y2_train), (x2_test, y2_test), c2 = src2
    print(x1_train.shape, y1_train.shape, x1_test.shape, y1_test.shape)
    print(x2_train.shape, y2_train.shape, x2_test.shape, y2_test.shape)
    x_train = np.vstack((x1_train, x2_train))
    y_train = np.vstack((y1_train, y2_train))
    x_test = np.vstack((x1_test, x2_test))
    y_test = np.vstack((y1_test, y2_test))
    print('Data combined to:')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test), c1+c2


def build_net(training_data, width=28, height=28):
    _, _, num_classes = training_data
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


def train(model, training_data, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

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
    print('EMNIST data loaded ' + str(emnist_data[2]) + ' classes')
    hasy_data = load_hasy_data()
    print('Hasy data loaded ' + str(hasy_data[2]) + ' classes')
    training_data = combine_data(emnist_data, hasy_data)
    print('Merged training data. Building and training neural network...')

    model = build_net(training_data)
    train(model, training_data, epochs=args.epochs)
