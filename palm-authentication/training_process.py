import urllib.request
import cv2
import numpy as np
import time
import os
from convert_img import Convert
from training import training_image
url = 'http://192.168.1.6/800x600.jpg'
im = None

count = 1
num_hands = 50
print("Pretraining!!!")
while True:
    name = input("Enter your name: ")
    output_path = 'E:/Palm-Authentication/train_images/' + name
    isExist = os.path.exists(output_path)
    if(isExist == False):
        break
# output_path = 'E:/Palm-Authentication/train_images/Hoang'
print(output_path)
raw_path = output_path + '/raw'
processed_path = output_path + '/processed'
print(raw_path)
print(processed_path)
os.makedirs(output_path)
os.makedirs(raw_path)
os.makedirs(processed_path)
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
start = time.time()
while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)
    cv2.imshow('live transmission', im)
    if(count > num_hands):
        break
    if(start + 1) <= time.time():
        print(time.time())
        cv2.imwrite(os.path.join(raw_path, name +'_'+ str(count) + '.jpg'), im)
        count = count + 1
        start = time.time()
    
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("Trainning")

for file in os.listdir(raw_path):
    convert = Convert(raw_path + '/' + file)
    print(raw_path + file)
    # print(convert.img)
    proc_img = convert.convert_img()
    cv2.imwrite(os.path.join(processed_path, file), proc_img)

from training import training_image
training = training_image(num_hands)