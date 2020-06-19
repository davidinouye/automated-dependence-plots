import numpy as np
import os, sys
import pickle
import scipy as scp
import matplotlib.pyplot as plt
import argparse
from scipy import misc
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from .data_utils_sign import *

# Define the classifier
def cnn_model(IMG_SIZE, NUM_CLASSES):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model

def train(model, x_train, y_train, lr=0.01, batch_size=16, epochs=30, model_name='traffic_model', load=False):
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    def lr_schedule(epoch):
        return lr * (0.1 ** int(epoch / 10))

    if not load:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                callbacks=[LearningRateScheduler(lr_schedule),
                            ModelCheckpoint('%s.h5'%model_name, save_best_only=True)])
    else:
        model.load_weights('%s.h5'%model_name)

    return model

def test(model, x_test, y_test):
    y_pred = model.predict_classes(x_test)
    acc = np.sum(y_pred == y_test) / np.size(y_pred)
    print("Test accuracy = {}".format(acc))
    return acc

def main(args):
    print('Loading Data...')
    if not args.aug:
        x_train, y_train, x_test, y_test = get_data(name='GTSRB')
    else:
        x_train, y_train, x_test, y_test = augment_data(name='GTSRB')

    # one-hot targets
    NUM_CLASSES = np.unique(y_train).shape[0]
    targets = np.eye(NUM_CLASSES, dtype='uint8')[y_train]

    print('Training...')
    model = cnn_model(IMG_SIZE, NUM_CLASSES)
    if args.aug == 'blur':
        model = train(model, x_train, targets, model_name=os.path.join(args.dir, 'traffic_model-aug-blur'), load=False)
    elif args.aug == 'bright':
        model = train(model, x_train, targets, model_name=os.path.join(args.dir, 'traffic_model-aug-bright'), load=False)
    elif args.aug == 'contrast':
        model = train(model, x_train, targets, model_name=os.path.join(args.dir, 'traffic_model-aug-contrast'), load=False)
    else:
        model = train(model, x_train, targets, model_name=os.path.join(args.dir, 'traffic_model'), load=False)

    print('Testing...')
    test_acc = test(model, x_test, y_test)

if __name__ == '__main__':
    if not os.path.isdir('models'):
        os.mkdir('models')
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, default='none')
    parser.add_argument('--data', type=str, default='GTSRB')
    parser.add_argument('--dir', type=str, default='models')
    args = parser.parse_args()
    main(args)
