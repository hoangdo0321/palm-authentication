import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import h5py
import os
from keras.preprocessing.image import ImageDataGenerator



train = ImageDataGenerator(rescale=1/255)
train_dataset = train.flow_from_directory('dataset/', target_size=(600, 600),color_mode='grayscale',class_mode = 'binary', batch_size=5)
print(train_dataset.class_indices)
print(train_dataset.classes)
print(train_dataset)


# data_list = []
# batch_index = 0

# while batch_index <= train_dataset.batch_index:
#     data = train_dataset.next()
#     data_list.append(data[0])
#     batch_index = batch_index + 1

# # now, data_array is the numeric data of whole images
# data_array = np.asarray(data_list)
# print(data_array)


x_train=np.concatenate([train_dataset .next()[0] for i in range(train_dataset .__len__())])
y_train=np.concatenate([train_dataset .next()[1] for i in range(train_dataset .__len__())])

model = keras.Sequential([
   keras.layers.Flatten(input_shape=(600, 600)), # dimensions of the image
   keras.layers.Dense(64, activation=tf.nn.relu),
   keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.summary()
# model = keras.Sequential([
#    keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(600, 600)),
#    keras.layers.MaxPool2D(2,2),
#    keras.layers.Conv2D(32, (3,3), activation = 'relu'),
#    keras.layers.MaxPool2D(2,2),
#    keras.layers.Conv2D(54, (3,3), activation = 'relu'),
#    keras.layers.MaxPool2D(2,2), 
#    keras.layers.Flatten(), # dimensions of the image
#    keras.layers.Dense(64, activation=tf.nn.relu),
#    keras.layers.Dense(1, activation=tf.nn.softmax)
# ])

model.compile(optimizer=tf.optimizers.Adam() ,
       loss='binary_crossentropy',
       metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(x_train, y_train)
print('Test accuracy:', test_acc)

# model.save("saved/neunet.h5")

