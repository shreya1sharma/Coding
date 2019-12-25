#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 00:57:43 2017

@author: saror
"""

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.models import Sequential, Model
from keras.utils import np_utils
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test

def sequential_model(x_train, y_train, x_test, y_test):
    model= Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense({{choice([256,512,1024])}}))
    model.add(Activation({{choice(['relu','sigmoid'])}}))
    model.add(Dropout({{uniform(0,1)}}))
    if conditional({{choice(['three','four'])}}) == 'four':
        model.add(Dense(100))
        model.add({{choice([Dropout(0.5),Activation('linear')])}})
        model.add(Activation('relu'))
        
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics= ['accuracy'], optimizer={{choice(['rmsprop','adam','sgd'])}})
    model.fit(x_train, y_train,
              batch_size={{choice([64,128])}},
              epochs=1,
              verbose=2,
              validation_data= (x_test, y_test))
    score, acc= model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss':-acc, 'status': STATUS_OK, 'model':model}

def functional_model(x_train, y_train, x_test, y_test):
    input_shape= 784
    input_img= Input(shape=(input_shape,))
    hidden_1= Dense(512, activation='relu')(input_img)
    hidden_2= Dropout({{uniform(0,1)}})(hidden_1)
    hidden_3= Dense({{choice([256,512,1024])}})(hidden_2)
    hidden_4= Activation({{choice(['relu','sigmoid'])}})(hidden_3)
    hidden_5= Dropout({{uniform(0,1)}})(hidden_4)
    hidden_6= Dense(10)(hidden_5)
    output_class=Activation('softmax')(hidden_6)
    model= Model(input_img, output_class)
    
    model.compile(loss='categorical_crossentropy', metrics= ['accuracy'], optimizer='adam')
    model.fit(x_train, y_train,
              batch_size={{choice([64,128])}},
              epochs=1,
              verbose=2,
              validation_data= (x_test, y_test))
    score, acc= model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss':-acc, 'status': STATUS_OK, 'model':model}
    

if __name__=='__main__':
     X_train, Y_train, X_test, Y_test = data()
     best_run, best_model= optim.minimize(model= functional_model,
                                          data=data,
                                          algo= tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
     print("evaluation of best performing model:", best_model.evaluate(X_test, Y_test))
     print("best performing model hyperparameters:", best_run)















