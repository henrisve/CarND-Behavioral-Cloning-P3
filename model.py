import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split


print("2")

#use pandas instead?
aws=True
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
    samples=np.concatenate([samples,samples2,samples3,samples5,samples6])


all_samples=[]

for line in samples:


    for camera in range(3):
        steering = float(line[3])
        image_file = line[camera].split('/')[-1]
        image_path = 'data/IMG/' + image_file
        correction = 0 if camera == 0 else (0.2 if camera == 1 else -0.2)
        steering += correction
        all_samples.append([image_path,steering,False])
        all_samples.append([image_path,-steering,True])

print(np.shape(all_samples))
train_samples, validation_samples = train_test_split(all_samples, test_size=0.2)



def rgb_clahe(img):
    #https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.merge((cl, a, b))


def warp_image(image, angle):
    shape = np.shape(image)
    zoomx = np.random.uniform(0.8, 1.2)
    # zoomxy=np.random.uniform(-0.2,0.2)
    zoomy = np.random.uniform(0.8, 1.2)  # zoomx-zoomxy
    zoomxy = zoomx - zoomy
    extrapixelsx = (shape[0] - shape[0] * zoomx) / 2
    extrapixelsy = (shape[1] - shape[1] * zoomy) / 2
    movex = np.random.uniform(-30, 30)
    movey = np.random.uniform(-15, 15)
    posx = extrapixelsx + movex / (zoomx * 2)
    posy = extrapixelsy + movey / (zoomy * 2)
    skewx = np.random.uniform(-0.1, 0.1)
    skewy = np.random.uniform(-0.05, 0.05)  ##take it easy with this!
    # print(f"zoomx {zoomx},zoomy {zoomy},movex {movex}, movey {movey}, skewx {skewx}, skewy {skewy}")
    M = np.float32([[zoomx, skewx, posx], [skewy, zoomy, posy]])
    image = cv2.warpAffine(image, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REPLICATE)
    angle += (skewx * 0.5) + (skewy * 0.5) + movex / 100 + zoomxy * angle
    return image, angle


def change_colors_image(image):
    image = image.astype(np.int32)
    sigma = 0  # 30 ##difference between channels
    mu = np.random.randint(-100, 100)  # brightness
    rnds = np.round(np.random.normal(mu, sigma, 3)).astype(int)
    for i, r in enumerate(rnds):
        if np.random.rand() < 0.1:
            image[:, :, i] = 255 - image[:, :, i]

        image[:, :, i] = np.clip(image[:, :, i] + r, 0, 255)

    return image.astype(np.uint8)


def dropout_image(image):
    shape = np.shape(image)
    # https://github.com/aleju/imgaug
    # Randomly remove parts of the image
    # Similar to dropout in the network, if the network can learn
    # on more limited parts, it should work better!
    for i in range(3):
        for j in range( np.random.randint(2, 15)):
            center = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
            size = (np.random.randint(0, shape[0] / 3), np.random.randint(0, shape[1] / 3))

            pt1 = (np.clip(center[0] - size[0], 0, shape[0]), np.clip(center[1] - size[1], 0, shape[1]))
            pt2 = (np.clip(center[0] + size[0], 0, shape[0]), np.clip(center[1] + size[1], 0, shape[1]))
            image[pt1[0]:pt2[0], pt1[1]:pt2[1], i] = 0

    return image

def augument_image(image, angle):
    if np.random.rand() < 0.5:
        image, angle = warp_image(image, angle)
    if np.random.rand() < 0.5:
        image = change_colors_image(image)
    if np.random.rand() < 0.5:
        image=dropout_image(image)
    return image, angle

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
                image = cv2.imread(center_path)
                shape = np.shape(image)
                #image = cv2.resize(image[int(shape[0]*0.25):int(shape[0]*0.85),0:shape[1],:],(200,66))
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)#YUV)
                #image = rgb_clahe(image)
                #image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)  # YUV)
                steering = batch_sample[1]
                if batch_sample[2]:  # flip image and steering, so we have 50/50 for both left and right
                    image = cv2.flip(image, 1)

                image, steering = augument_image(image, steering)

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
#model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1. , input_shape=(66,200,3)))
model.add(Conv2D(34, (5, 5), activation="relu", strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.8, noise_shape=None, seed=None))
model.add(Conv2D(56, (5, 5), activation="relu"))
#model.add(MaxPooling2D())
model.add(Conv2D(78, (3, 3), activation="relu", strides=(2, 2)))
#model.add(MaxPooling2D())
model.add(Conv2D(84, (3, 3), activation="relu"))
model.add(Conv2D(104, (3, 3), activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1000))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(500))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(100))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(50))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(10))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=2)
model.save('model1.h5')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=1)
model.save('model2.h5')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=1)
model.save('model3.h5')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=1)
model.save('model4.h5')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=1)
model.save('model5.h5')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=1)
model.save('model6.h5')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
                    validation_steps=len(validation_samples)//32, nb_epoch=1)
model.save('model7.h5')
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
#                    nb_val_samples=len(validation_samples), nb_epoch=3)


