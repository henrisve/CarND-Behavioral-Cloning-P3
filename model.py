import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split

print("1")

#use pandas instead?
with open('data/driving_log.csv', 'r') as file:
    samples = [line for line in csv.reader(file, delimiter=',')][1:]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# todo, what aobut saving the data to file, and then...
# use the generator to take lines from that file
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                camera=np.random.randint(3)
                flip=np.random.randint(2)
                center_file = batch_sample[camera].split('/')[-1]
                center_path = 'data/IMG/' + center_file
                center_image = cv2.cvtColor(cv2.imread(center_path), cv2.COLOR_BGR2RGB)
                correction = 0 if camera == 0 else (0.2 if camera == 1 else -0.2)
                steering = float(batch_sample[3])+correction
                if flip: # flip image and steering, so we have 50/50 for both left and right
                    center_image=cv2.flip(center_image,1)
                    steering=-steering

                images.append(center_image)
                angles.append(steering)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# the training part, maybe split into two files?

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

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
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
