import csv
import numpy as np
import cv2


with open('data/driving_log.csv', 'r') as file:
    lines = [line for line in csv.reader(file, delimiter=',')][1:]
images=[]
measurements=[]
for line in lines:
    #todo, make nicer
    center_file = line[0].split('/')[-1]
    left_file = line[1].split('/')[-1]
    right_file = line[2].split('/')[-1]
    center_path = 'data/IMG/' + center_file
    left_path = 'data/IMG/' + left_file
    right_path = 'data/IMG/' + right_file
    center_image=cv2.imread(center_path)
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)
    steering=float(line[3])
    correction = 0.2
    images.append(center_image)
    measurements.append(steering)
    images.append(left_image)
    measurements.append(steering+correction)
    images.append(right_image)
    measurements.append(steering-correction)

#what about having np array to start with instead of converting?

X_train=np.array(images)

y_train=np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D

model= Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 -0.5))#, input_shape=(160,320,3)))
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model.h5')
