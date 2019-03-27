import argparse
import os
import pickle

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
    training_images = training_images[:, 2:30, 2:30, :].reshape(max_, 28, 28, 1)
    training_labels = data['y_train'][:max_]

    max_ = data['x_test'].shape[0]
    testing_images = (data['x_test'][:max_]).reshape(max_, 32, 32, 1)
    testing_images = testing_images[:, 2:30, 2:30, :].reshape(max_, 28, 28, 1)
    testing_labels = data['y_test'][:max_]

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    return (training_images, training_labels), (testing_images, testing_labels), len(data['labels'])


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


def train(model, emnist, hasy, num_classes, batch_size=256, epochs=5):
    (x_train, y_train), (x_test, y_test), _ = emnist

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    print('Training on EMNIST data')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print('Training on Hasy data')
    (x_train, y_train), (x_test, y_test), _ = hasy
    print(type(x_train), type(y_train), type(x_test), type(y_test))
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    print(type(x_train), type(y_train), type(x_test), type(y_test))

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
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train on')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    emnist_data = load_emnist_data(args.file)
    print('EMNIST data loaded ' + str(emnist_data[2]) + ' classes')
    hasy_data = load_hasy_data()
    print('Hasy data loaded ' + str(hasy_data[2]) + ' classes')

    num_classes = emnist_data[2] + hasy_data[2]
    model = build_net(num_classes)
    train(model, emnist_data, hasy_data, num_classes, epochs=args.epochs)
