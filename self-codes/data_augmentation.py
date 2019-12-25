#Keras Documentation: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#Jason Brownlee: http://machinelearningmastery.com/image-augmentation-deep-learning-keras/

import tensorflow
import keras
#import scipy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import PIL
from PIL import Image

'''#defining data augmentation
datagen= ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode= 'nearest')
#loading an image
img= Image.open()
img.load()


x= img_to_array(img)
x= x.reshape((1,)+x.shape)

datagen.fit(x)
i=0
for batch in datagen.flow(x, batch_size=1, save_to_dir = 'D:\Codes\self-study\preview', save_prefix = 'cat', save_format='jpeg'):
    i+=1
    if i>20:
        break'''
    
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = "D:/Codes/datasets/dogs-cats/data/train"
validation_data_dir = "D:/Codes/datasets/dogs-cats/data/validation"
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model= Sequential()
model.add(Conv2D(32,(3,3), input_shape= input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer= 'rmsprop', metrics=['accuracy'])

#data augmentaion using batches of data direcctky from directory

train_datagen=  ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(150,150), batch_size= batch_size, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(150,150), batch_size= batch_size, class_mode='binary')


model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
        
model.save_weights('first_try.h5py') 



