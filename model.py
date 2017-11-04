import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split


print("2")

#use pandas instead?
aws=False
with open('data/driving_log.csv', 'r') as file:
    samples = [line for line in csv.reader(file, delimiter=',')][1:]
if aws:
    with open('data/driving_log2.csv', 'r') as file:
        samples2 = [line for line in csv.reader(file, delimiter=',')][1:]

    with open('data/drivinglog_map2.csv', 'r') as file:
        samples3 = [line for line in csv.reader(file, delimiter=',')][1:]

    with open('data/drivinglog_dirt.csv', 'r') as file:
        samples4 = [line for line in csv.reader(file, delimiter=',')][1:]

    with open('data/driving_log_left.csv', 'r') as file:
        samples5 = [line for line in csv.reader(file, delimiter=',')][1:]

    with open('data/driving_log_right.csv', 'r') as file:
        samples6 = [line for line in csv.reader(file, delimiter=',')][1:]
    samples=np.concatenate([samples,samples2,samples3,samples4,samples5,samples6])


all_samples=[]

for line in samples:
    steering = float(line[3])

    for camera in range(1):
        image_file = line[camera].split('/')[-1]
        image_path = 'data/IMG/' + image_file
        correction = 0 if camera == 0 else (0.2 if camera == 1 else -0.2)
        steering += correction
        all_samples.append([image_path,steering,False])
        all_samples.append([image_path,-steering,True])

print(np.shape(all_samples))
train_samples, validation_samples = train_test_split(all_samples, test_size=0.2)

def distort_image(image):
    return image

def rgb_clahe(img):
    #https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.merge((cl, a, b))



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.preprocessing.image import *

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                image = cv2.cvtColor(cv2.imread(center_path), cv2.COLOR_BGR2LAB)#YUV)
                image = rgb_clahe(image)
                image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)  # YUV)
                #cv2.imshow('image', image)
                #print("2")
                #cv2.waitKey(0)
                steering = batch_sample[1]
                if batch_sample[2]:  # flip image and steering, so we have 50/50 for both left and right
                    image = cv2.flip(image, 1)

                #image = array_to_img(image)
                #image=random_rotation(image,20)
                #print("3")
                #cv2.waitKey(0)
                #image=random_shift(image,.1,.1)
                #cv2.imshow('image', image)
                #print("4")
                #cv2.waitKey(0)
                #image=random_shear(image,.1)
                #cv2.imshow('image', image)
                #print("5")
                #cv2.waitKey(0)
                ##image=random_zoom(image,(1,1))
                #cv2.imshow('image', image)
                #print("6")
                #cv2.waitKey(0)
                #image = img_to_array(image)
                images.append(image)
                angles.append(steering)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# the training part, maybe split into two files?





model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.))  # , input_shape=(160,320,3)))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.8, noise_shape=None, seed=None))
model.add(Conv2D(36, (5, 5), activation="relu"))
#model.add(MaxPooling2D())
model.add(Conv2D(48, (3, 3), activation="relu", strides=(2, 2)))
#model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(50))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(10))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=5)

#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
#                    nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
