import csv
import cv2
import sys

import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Dropout
from keras.layers.convolutional import Cropping2D
from sklearn.model_selection import train_test_split


if(len(sys.argv) < 2):
    epochs = 4
else:
    epochs = int(sys.argv[1])
print(epochs)

def flipImages(images,labels):

    flipped = []
    measurement = []

    for i in range(len(images)):
        flipped.append(cv2.flip(images[i],1))
        measurement.append(-labels[i])

    return flipped,measurement


lines = []
with open('data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

images = [[],[],[]]

measurements = []
for line in lines:
    fail = False
    for i in range(0,3):
        if i > 0 :
            continue
        filename = line[i].split("\\")[-1]
        img = cv2.imread('data/IMG/'+filename)
        if img is None:
            fail = True
            print(filename)
            break
        images[i].append(img)

    if not fail:
        measurements.append(float(line[3]))

images_flip,flip_measurement = flipImages(images[0],measurements)

all_images = np.concatenate((images[0],images_flip))
X_train = all_images
all_measurements = np.concatenate((measurements,flip_measurement))
y_train = np.array(all_measurements)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0)-0.5))
model.add(Conv2D(24, (5, 5), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=epochs)

model.save(r'model.h5')

