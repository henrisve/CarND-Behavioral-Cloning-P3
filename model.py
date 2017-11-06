import csv
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split



with open('data/driving_log.csv', 'r') as file:
    samples = [line for line in csv.reader(file, delimiter=',')][1:]
if True: ## set to false to only take one dataset
    with open('data/driving_log2.csv', 'r') as file:
        samples2 = [line for line in csv.reader(file, delimiter=',')][1:]

    with open('data/drivinglog_map2.csv', 'r') as file:
        samples3 = [line for line in csv.reader(file, delimiter=',')][1:]

    #with open('data/drivinglog_dirt.csv', 'r') as file:
    #    samples4 = [line for line in csv.reader(file, delimiter=',')][1:]

    with open('data/driving_log_left.csv', 'r') as file:
        samples5 = [line for line in csv.reader(file, delimiter=',')][1:]

    with open('data/driving_log_right.csv', 'r') as file:
        samples6 = [line for line in csv.reader(file, delimiter=',')][1:]
    samples=np.concatenate([samples,samples2,samples3,samples5,samples6])

''' Go through  all the samples.
    makes a list of all the images and steering, also
    creates more data for the turns, so it will become more equal
'''
all_samples=[]
for line in samples:
    for camera in range(3):
        steering = float(line[3])
        if True:  # steering != 0 or np.random.rand() <0.3: #Remove may cause the car to go less straight
            image_file = line[camera].split('/')[-1]
            image_path = 'data/IMG/' + image_file
            correction = 0 if camera == 0 else (0.2 if camera == 1 else -0.2)
            #  steering += correction

            for i in range(1):# + int((abs(steering)*3.)**1.2)):
                all_samples.append([image_path, steering+correction,False])
                all_samples.append([image_path, -(steering+correction),True])

print(np.shape(all_samples))
train_samples, validation_samples = train_test_split(all_samples, test_size=0.2)


'''Adds a "clahe" normalizantion to the image
#https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
'''
def rgb_clahe(img):
    #https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.merge((cl, a, b))

'''Warp images
    This function will warp, zoom and move the image
    Used to get more data'''
def warp_image(image, angle):
    shape = np.shape(image)
    zoomx = np.random.uniform(0.8, 1.2)
    zoomy = np.random.uniform(0.8, 1.2)  # zoomx-zoomxy
    zoomxy = zoomx - zoomy
    #cv2 zoom to the corner, so need to move to center.
    extrapixelsx = (shape[0] - shape[0] * zoomx) / 2
    extrapixelsy = (shape[1] - shape[1] * zoomy) / 2
    movex = np.random.uniform(-30, 30)
    movey = np.random.uniform(-15, 15)
    posx = extrapixelsx + movex / (zoomx * 2)
    posy = extrapixelsy + movey / (zoomy * 2)
    skewx = np.random.uniform(-0.1, 0.1)
    skewy = np.random.uniform(-0.05, 0.05)
    M = np.float32([[zoomx, skewx, posx], [skewy, zoomy, posy]])
    image = cv2.warpAffine(image, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REPLICATE)
    #Some changes will affect how we "should" turn for that image
    angle += (skewx * 0.5) + (skewy * 0.5) + movex / 100 + zoomxy * angle
    return image, angle


''' Change color of images.
    This function will randomly brigthen/darken images.
    it can also do this different for each channel, and thus change the overall color
'''
def change_colors_image(image):
    image = image.astype(np.int32)  #to not allow values to go from 255 to 0.
    sigma = 20  # 30 ##difference between channels
    mu = np.random.randint(-100, 100)  # brightness
    rnds = np.round(np.random.normal(mu, sigma, 3)).astype(int)
    for i, r in enumerate(rnds):
        if np.random.rand() < 0.1:
            image[:, :, i] = 255 - image[:, :, i]

        image[:, :, i] = np.clip(image[:, :, i] + r, 0, 255)

    return image.astype(np.uint8)


''' Image dropout
    got the idea from https://github.com/aleju/imgaug
    Randomly remove parts of the image
    Similar to dropout in the network, if the network can learn
    on more limited parts, it should generalize better!
    It works by choosing a center point randomly, and then draw
    a rectangle with random size.
'''
def dropout_image(image):
    shape = np.shape(image)

    for i in range(3):
        for j in range( np.random.randint(1, 15)):
            center = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
            size = (np.random.randint(0, shape[0] / 3), np.random.randint(0, shape[1] / 3))

            pt1 = (np.clip(center[0] - size[0], 0, shape[0]), np.clip(center[1] - size[1], 0, shape[1]))
            pt2 = (np.clip(center[0] + size[0], 0, shape[0]), np.clip(center[1] + size[1], 0, shape[1]))
            image[pt1[0]:pt2[0], pt1[1]:pt2[1], i] = 0

    return image


''' Augment images
    This function will call the other
    function that will change images so we get new data
'''
def augument_image(image, angle):
    if np.random.rand() < 0.8:
        image, angle = warp_image(image, angle)
    if np.random.rand() < 0.8:
        image = change_colors_image(image)
    if np.random.rand() < 0.8:
        image=dropout_image(image)
    return image, angle



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.preprocessing.image import *


''' Generator
    generates data for training and validation
    it loads the images from disk, if for training
    we also augment it so we get new data each time
'''
def generator(samples, train=True,batch_size=32):
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

                steering = batch_sample[1]
                if batch_sample[2]:  # flip image and steering, so we have 50/50 for both left and right
                    image = cv2.flip(image, 1)

                if train:
                    image, steering = augument_image(image, steering)

                images.append(image)
                angles.append(steering)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples,train=True)
validation_generator = generator(validation_samples,train=False)


''' Model starts here'''
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1. , input_shape=(66,200,3)))
model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Conv2D(36, (5, 5), activation="elu"))
#model.add(MaxPooling2D())
model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
#model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
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
                    validation_steps=len(validation_samples)//32, nb_epoch=2)
model.save('model1.h5')
#model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
#                    validation_steps=len(validation_samples)//32, nb_epoch=1)
#model.save('model2.h5')
#model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
#                    validation_steps=len(validation_samples)//32, nb_epoch=1)
#model.save('model3.h5')
#model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
#                    validation_steps=len(validation_samples)//32, nb_epoch=1)
#model.save('model4.h5')
#model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
#                    validation_steps=len(validation_samples)//32, nb_epoch=1)
#model.save('model5.h5')
#model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
#                    validation_steps=len(validation_samples)//32, nb_epoch=1)
##model.save('model6.h5')
#model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32, validation_data=validation_generator,
#                    validation_steps=len(validation_samples)//32, nb_epoch=1)
##model.save('model7.h5')
#This could be in a for loop, but just here temporarly so I can compare the epochs i the simulator.


#
