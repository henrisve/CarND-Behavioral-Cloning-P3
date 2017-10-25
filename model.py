import csv
import numpy as np
import cv2


with open('data/driving_log.csv', 'r') as file:
    lines = [line for line in csv.reader(file, delimiter=',')][1:]
images=[]
measurements=[]
for line in lines:
    filename = line[0].split('/')[-1]

    current_path = 'data/IMG/' + filename
    image=cv2.imread(current_path)

    images.append(image)

    measurements.append(float(line[3]))

#what about having np array to start with instead of converting?

X_train=np.array(images)

y_train=np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D

model= Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5)

model.save('model.h5')
