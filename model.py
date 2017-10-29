import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
with open('data/driving_log.csv', 'r') as file:
    samples = [line for line in csv.reader(file, delimiter=',')][1:]



train_samples, validation_samples = train_test_split(samples, test_size=0.2)



#todo, what aobut saving the data to file, and then...
# use the generator to take lines from that file
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_file = line[0].split('/')[-1]
                left_file = line[1].split('/')[-1]
                right_file = line[2].split('/')[-1]
                center_path = 'data/IMG/' + center_file
                left_path = 'data/IMG/' + left_file
                right_path = 'data/IMG/' + right_file
                center_image = cv2.cvtColor(cv2.imread(center_path), cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
                steering = float(line[3])
                correction = 0.2
                images.append(center_image)
                angles.append(steering)
                images.append(left_image)
                angles.append(steering - correction)
                #Flip images to get more data
                #todo check this is correct
                images.append(cv2.flip(center_image))
                angles.append(steering + correction)
                images.append(right_image)
                angles.append(-steering)
                images.append(cv2.flip(left_image))
                angles.append(-steering - correction)
                images.append(cv2.flip(right_image))
                angles.append(-steering + correction)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#the training part, maybe split into two files?

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda , Dropout
from keras.layers import Conv2D, MaxPooling2D,Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.))  # , input_shape=(160,320,3)))
model.add(Conv2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(84))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
