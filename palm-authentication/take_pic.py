import cv2
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import concurrent.futures
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from convert_img import Convert

url = 'http://192.168.43.82/800x600.jpg'
im = None
output_path = 'E:/Palm-Authentication/test_images'
count = 0
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

cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)
    cv2.imshow('live transmission', im)
    # print(im)
    cv2.imwrite(os.path.join('E:/Palm-Authentication/test_images', 'test_pic.jpg'), im)
    conv = Convert('E:/Palm-Authentication/test_images/test_pic.jpg')
    pic = conv.convert_img()
    print(type(pic))
    count = count + 1
    model = keras.models.load_model("saved/neunet.h5")
    res_pic = np.array([np.array(pic)])
    res_pic = res_pic / 255.0
    prediction = model.predict(res_pic)
    # print((prediction > 0.5).astype("int32"))
    print(np.max(prediction))
    print(classes[np.argmax(prediction)])
    if count == 100:
        break
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
