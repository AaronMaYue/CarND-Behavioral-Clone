import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Convolution2D, Lambda, Cropping2D, Dropout
from keras.models import Sequential
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time


def open_file(file_path):
    samples = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for line in reader:
            samples.append(line)
    return samples

# reframe path and steering data


def define_path(samples, path_name):
    path = []
    angles = []
    for sample in samples:
        c_name = path_name + sample[0].split('/')[-1]
        path.append(c_name)
        l_name = path_name + sample[1].split('/')[-1]
        path.append(l_name)
        r_name = path_name + sample[2].split('/')[-1]
        path.append(r_name)

        steer = float(sample[3])
        angles.append(steer)
        angles.append(steer + 0.2)
        angles.append(steer - 0.2)
    return path, angles



# import data path
data1_csv_path = './data1/driving_log.csv'
data2_csv_path = './data2/driving_log.csv'


data1_samples = open_file(data1_csv_path)
data2_samples = open_file(data2_csv_path)

data1_file_name = './data1/IMG/'
data2_file_name = './data2/IMG/'

data1_imgs_path, data1_angles = define_path(data1_samples[1:], data1_file_name)
data2_imgs_path, data2_angles = define_path(data2_samples, data2_file_name)

# left and right steering correctin
data_imgs_paths = data1_imgs_path + data2_imgs_path
data_angles = data1_angles + data2_angles

# augment data
aug_paths = []
aug_angles = []
aug_flags = []

for i in range(len(data_imgs_paths)):
    if np.random.rand() < 0.5:
        aug_paths.append(data_imgs_paths[i])
        aug_angles.append(data_angles[i])
        aug_flags.append(0)

        aug_paths.append(data_imgs_paths[i])
        aug_angles.append(data_angles[i] * -1)
        aug_flags.append(1)

    else:
        aug_paths.append(data_imgs_paths[i])
        aug_angles.append(data_angles[i])
        aug_flags.append(0)

# random data
shuf_paths, shuf_angles, shuf_flags = shuffle(aug_paths, aug_angles, aug_flags)


# split data into training data and validation data
train_index = int(len(shuf_paths) * 0.8)
train_datas = (shuf_paths[:train_index], shuf_angles[:train_index], shuf_flags[:train_index])
valid_datas = (shuf_paths[train_index:], shuf_angles[train_index:], shuf_flags[train_index:])

# generator


def generator(datas, batch_size):
    num_examples = len(datas[0])
    while True:
        for offset in range(0, num_examples, batch_size):
            batch_paths = datas[0][offset:offset + batch_size]
            batch_angles = datas[1][offset:offset + batch_size]
            batch_flags = datas[2][offset:offset + batch_size]

            images = []
            angles = []
            for path, angle, flag in zip(batch_paths, batch_angles, batch_flags):
                img = mpimg.imread(path)
                if flag == 1:
                    images.append(cv2.flip(img, 1))
                    angles.append(angle)
                else:
                    images.append(img)
                    angles.append(angle)
            batch_x = np.array(images)
            batch_y = np.array(angles)
            yield shuffle(batch_x, batch_y)


BATCH_SIZE = 512
train_generator = generator(train_datas, BATCH_SIZE)
valid_generator = generator(valid_datas, BATCH_SIZE)


# model architecture
activation = 'relu'
t1 = time.time()
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
#model.add(Lambda(lambda x: x/255.0))
model.add(Cropping2D(cropping=((75, 20), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation=activation))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation=activation))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation=activation))
model.add(Convolution2D(64, 3, 3, activation=activation))
model.add(Convolution2D(64, 3, 3, activation=activation))
model.add(Flatten())
model.add(Dropout(0.5))
# model.add()
model.add(Dense(100, activation=activation))
# model.add(Dropout(0.5))
model.add(Dense(50, activation=activation))
model.add(Dense(10, activation=activation))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split = 0.2, nb_epoch = 2, shuffle=False)
model.fit_generator(train_generator, samples_per_epoch=len(train_datas[0]), validation_data=valid_generator,
                    nb_val_samples=len(valid_datas[0]), nb_epoch=3)
t2 = time.time()
t = t2 - t1
model.save('model.h5')
print('Training Finish!')
print('Time: ', t)
