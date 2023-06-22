import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import h5py

num_right_train = 20
num_left_train = 20
classes = ["left", "right"]
data_folder = 'E'
#1 for right, 0 for left
pic = np.array(Image.open("train_images/ibrahim_right_" + str(0) + ".jpg"))
train_labels = np.array([1]*num_right_train + [0]*num_left_train)
train_images = np.array([pic])
for i in range(1, num_right_train):
   pic = np.array(Image.open("train_images/ibrahim_right_" + str(i) + ".jpg"))
   train_images = np.vstack((train_images, np.array([pic])))
for i in range(num_left_train):
   pic = np.array(Image.open("train_images/ayush_right_" + str(i) + ".jpg"))
   train_images = np.vstack((train_images, np.array([pic])))
train_images = train_images / 255.0

# train_images = tf.stack(train_images)
# train_labels = tf.stack(train_labels)
print(train_images)
print(train_labels)

model = keras.Sequential([
   keras.layers.Flatten(input_shape=(600, 600)), # dimensions of the image
   keras.layers.Dense(64, activation=tf.nn.relu),
   keras.layers.Dense(1, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam() ,
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy'])

model.summary()

print(type(train_images))
print(type(train_labels))
model.fit(train_images, train_labels, epochs=10)
#model.fit(train_images, epochs=5)

# model.save("saved/neunet.h5")
