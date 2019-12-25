import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

from PIL import Image
from keras import backend as K
img_width, img_height= 28, 28
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
    
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32') / 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("sigmoid"))

model.add(Convolution2D(16, 5, 5, border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("sigmoid"))
model.add(Dropout(0.5))

model.add(Convolution2D(120, 1, 1, border_mode='valid'))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation('softmax'))


l_rate = 1
sgd = SGD(lr=l_rate, momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics= ['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=2, verbose=1, validation_data=(X_test, Y_test))

'''sgd = SGD(lr=0.8 * l_rate,  momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics= ['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=3, verbose=1,  validation_data=(X_test, Y_test))

sgd = SGD(lr=0.4 * l_rate,  momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metric= ['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=3, verbose=1, validation_data=(X_test, Y_test))

sgd = SGD(lr=0.2 * l_rate, momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics= ['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=4, verbose=1,  validation_data=(X_test, Y_test))

sgd = SGD(lr=0.08 * l_rate,  momentum=0.8)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics= ['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=8,  verbose=1,  validation_data=(X_test, Y_test))'''

scores= model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_hat = model.predict_classes(X_test)
test_wrong = [im for im in zip(X_test,y_hat,y_test) if im[1] != im[2]]

plt.figure(figsize=(10, 10))
for ind, val in enumerate(test_wrong[:100]):
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.subplot(10, 10, ind + 1)
    im = 1 - val[0].reshape((28,28))
    plt.axis("off")
    plt.text(0, 0, val[2], fontsize=14, color='blue')
    plt.text(8, 0, val[1], fontsize=14, color='red')
    plt.imshow(im, cmap='gray')
    

