import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
import keras
import matplotlib.pyplot as plt
lines=[]
csv_path=os.path.join(PROJECT_ROOT, 'data', 'driving_log.csv')
with open(os.path.join(PROJECT_ROOT, 'data', 'driving_log.csv')) as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                filename_center = batch_sample[0].split('/')[-1]
                filename_left = batch_sample[1].split('/')[-1]
                filename_right = batch_sample[2].split('/')[-1]
                current_path_center = os.path.join(PROJECT_ROOT, 'data', 'IMG', filename_center)
                current_path_left = os.path.join(PROJECT_ROOT, 'data', 'IMG', filename_left)
                current_path_right = os.path.join(PROJECT_ROOT, 'data', 'IMG', filename_right)
                image_center = cv2.imread(current_path_center)
                image_left = cv2.imread(current_path_left)
                image_right = cv2.imread(current_path_right)
                images.append(image_center)
                measurement_center = float(batch_sample[3])
                measurements.append(measurement_center)
                correction = 0.2
                images.append(image_left)
                measurement_left = measurement_center + correction
                measurements.append(measurement_left)
                images.append(image_right)
                measurement_right = measurement_center - correction
                measurements.append(measurement_right)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Lambda
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Cropping2D
model=Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
history_object=model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/32,
  validation_data=validation_generator, validation_steps=len(validation_samples),
  epochs=5,verbose=1)

model.save('model.h5')
print('Model saved succesfully')

# Plotting
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean s quared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()