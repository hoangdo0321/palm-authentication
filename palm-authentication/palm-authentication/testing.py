from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
classes = []
data_folder_path = 'E:/Palm-Authentication/train_images'

def update_classes(path):
        clas = []
        for folder in os.listdir(path):  
            print(folder)  
            clas.append(folder)
        return clas
classes = update_classes(data_folder_path)
print(classes)
im = Image.open("E:/palm-vein-door-unlock-master/data/processed/ibrahim_right_10.jpg") 
# im = Image.open("E:/Palm-Authentication/train_images/Hoang/processed/Hoang_9.jpg") 

im.show()
model = keras.models.load_model("saved/neunet.h5")
test_img = np.array([np.array(im)])
predictions=model.predict(test_img)
print(predictions)
# print(classes[np.argmax(predictions)])


# result = np.where(predictions > 0.5, 1, 0)
# print("Result: ", result)


# for prediction in predictions:
#     print(prediction)
#     if prediction == 0:
#         print("Left hand")
#     else:
#         print("right Hand")