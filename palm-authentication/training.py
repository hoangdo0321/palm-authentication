import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import h5py
import os


def training_image(hand_num):

    NUM_HAND_IMG = hand_num
    classes = []
    data_folder_path = 'E:/Palm-Authentication/train_images'
    saved_path = 'saved'
    def update_classes(path):
        clas = []
        for folder in os.listdir(path):  
            print(folder)  
            clas.append(folder)
        return clas
    classes = update_classes(data_folder_path)

    print(len(classes))

    train_labels = np.array([0] * NUM_HAND_IMG)

    for num in range(1, len(classes)):
        arr = np.array([num] * NUM_HAND_IMG)
        train_labels =np.concatenate((train_labels, arr))
    print(len(train_labels))

    pic = np.array(Image.open("train_images/" + classes[0] + "/processed/" + classes[0] + "_" +  str(1) + ".jpg"))
    print("train_images/" + classes[0] + "/processed/" + classes[0] + "_" +  str(1) + ".jpg")
    train_images = np.array([pic])
    for i in range(len(classes)):
        if(i == 0):
            sta = 1
        else: sta = 0
        for j in range(sta, NUM_HAND_IMG):
                pic = np.array(Image.open("train_images/" + classes[i] + "/processed/" + classes[i] + "_" +  str(j+1) + ".jpg"))
                print("train_images/" + classes[i] + "/processed/" + classes[i] + "_" +  str(j+1) + ".jpg")
                train_images =np.vstack((train_images, np.array([pic])))

    train_images = train_images / 255.0

    # print(train_labels)
    # print(train_images)

    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(600, 600)), # dimensions of the image
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(len(classes), activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.optimizers.Adam() ,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    print(type(train_images))
    print(type(train_labels))
    model.fit(train_images, train_labels,steps_per_epoch = 3, epochs=10, validation_split = 0.2)
    train_loss, train_acc = model.evaluate(train_images, train_labels)
    print("Accuracy: " , train_acc)
    try:
        os.remove(saved_path)
    except OSError:
        pass
    model.save("saved/neu.h5")

