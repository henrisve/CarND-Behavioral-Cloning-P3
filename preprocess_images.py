import cv2
import csv
import numpy as np

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

def rgb_clahe(img):
    #https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.merge((cl, a, b))

for line in samples:
    for camera in range(3):
        image_file = line[camera].split('/')[-1]
        image_path = 'data/IMG/' + image_file
        image=cv2.imread(image_path)
        cv2.imwrite('data/IMGold/' + image_file, image)
        shape=np.shape(image)
        image = cv2.resize(image[int(shape[0] * 0.25):int(shape[0] * 0.85), 0:shape[1], :], (200, 66))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)#YUV)
        image = rgb_clahe(image)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)  # YUV)
        cv2.imwrite('data/IMG/' + image_file, image)



