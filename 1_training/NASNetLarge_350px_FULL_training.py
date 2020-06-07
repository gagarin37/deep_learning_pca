#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""

#Import NASNetLarge architecture with imagenet weights
from keras.applications import NASNetLarge
conv_base = NASNetLarge(weights='imagenet', include_top=False, input_shape=(350,350,3))

from keras import models, layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))


#v0, convolutional layers 17-18 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_18' or layer.name == 'normal_conv_1_17':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

#Optimizer = Adam
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['acc'])

#folder with training dataset
train_dir = '/tolkach/training_dataset_SN/'

# Data generator
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

#Augmentation: horizontal and vertical flips
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip=True,
        vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(350,350),
        batch_size=100,
        class_mode='categorical')

#train for 7 Epochs
history = model.fit_generator(train_generator, epochs=7)

##Consecutive stepwise release of further convolutional layers
#v1, + layers 14-16 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_16':
        set_trainable = True
    if layer.name == 'normal_conv_1_15':
        set_trainable = True
    if layer.name == 'normal_conv_1_14':
        set_trainable = True

    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])
model.summary()

#train for 1 epoch
history = model.fit_generator(train_generator, epochs=1)

#save weights


#v2, + layers 11-13 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_13':
        set_trainable = True
    if layer.name == 'normal_conv_1_12':
        set_trainable = True
    if layer.name == 'normal_conv_1_11':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])
model.summary()

#train for 1 epoch
history = model.fit_generator(train_generator, epochs=1)

#save weights

#v3, +layers 9-10 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_9':
        set_trainable = True
    if layer.name == 'normal_conv_1_10':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])
model.summary()

history = model.fit_generator(train_generator, epochs=1)

#save weights


#v4, + layers 7-8 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_7':
        set_trainable = True
    if layer.name == 'normal_conv_1_8':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])
model.summary()

#train for 1 epoch
history = model.fit_generator(train_generator, epochs=1)

#save weights


#v5, + layers 5-6 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_5':
        set_trainable = True
    if layer.name == 'normal_conv_1_6':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])
model.summary()

#train for 1 epoch
history = model.fit_generator(train_generator, epochs=1)

#save weights

#v6, + layers 3-4 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_4':
        set_trainable = True
    if layer.name == 'normal_conv_1_3':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])
model.summary()

#train for 1 epoch
history = model.fit_generator(train_generator, epochs=1)

#save weights

#v7, + layers 1-2 free
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'normal_conv_1_2':
        set_trainable = True
    if layer.name == 'normal_conv_1_1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])
model.summary()

#train for 1 epoch
history = model.fit_generator(train_generator, epochs=1)


#save weights and model
model.save_weights('TvN_350_SN_D256_V8_Ep1.weights')
model.save('TvN_350_SN_D256_V8_Ep1.h5')