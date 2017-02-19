
# coding: utf-8

# In[1]:

# Behavioral Cloning 

import numpy as np
import math

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')


# In[2]:

import csv
import random

#samples=[]
reduced_samples = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #samples.append(line)
        reduced_samples.append(line)
#samples.pop(0)  # remove the header
reduced_samples.pop(0)
#random.shuffle(samples)
random.shuffle(reduced_samples)
#reduced_sample_size = int(len(samples)/2)  #reduce sample size by 2
#reduced_samples = []
#for i in range(0,reduced_sample_size):
    #reduced_samples.append(samples[i])
#print(len(samples))
print(len(reduced_samples))


# In[3]:

# using generator code described in Behviral Cloning, 10: Generator
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(reduced_samples, test_size=0.2)

import sklearn
import matplotlib.pyplot as plt
from numpy import array
import cv2

def generator(reduced_samples, batch_size=32):
    gen_batch_size = int(batch_size/4)
    num_samples = len(reduced_samples)  
    str_corr = 0.18  # steering correction, used for left/right camera images
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, gen_batch_size):
            batch_samples = reduced_samples[offset:offset+gen_batch_size]

            images = []
            steerings = []
            
            for batch_sample in batch_samples:
                for i in range(0,3):
                    path_name = 'data/' + (batch_sample[i].strip())
                    road_image = cv2.imread(path_name)
                    str_angle = float(batch_sample[3])
                    if i == 1:  # left turn
                        str_angle = float(batch_sample[3])+str_corr
                    elif i == 2: # right turn
                        str_angle = float(batch_sample[3])-str_corr
                    images.append(road_image)
                    steerings.append(str_angle)
                    if i == 0:
                        my_flipped_image = cv2.flip(road_image, 1)
                        str_angle_flipped = -1.0*float(str_angle)
                        images.append(my_flipped_image)
                        steerings.append(str_angle_flipped)
            
            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)
    
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=4)
validation_generator = generator(validation_samples, batch_size=4)


# In[4]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Convolution2D(24, 5, 5, border_mode='valid'))
print(model.output_shape)
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
#model.add(BatchNormalization())

model.add(Convolution2D(36, 5, 5, border_mode='valid'))
print(model.output_shape)
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
#model.add(BatchNormalization())

model.add(Convolution2D(48, 3, 3, border_mode='valid'))
print(model.output_shape)
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
#model.add(BatchNormalization())
          
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
print(model.output_shape)
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())
print(model.output_shape)
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(10))
print(model.output_shape)
model.add(Activation('relu'))
model.add(Dense(1))


# In[5]:

#model.compile(optimizer='adam', loss='mse')
#model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=5)
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,  
                    validation_data=validation_generator,samples_per_epoch=len(train_samples)*4,
                    nb_val_samples=len(validation_samples)*4, nb_epoch=5)


# In[6]:

model.save('model.h5')


# In[ ]:



