import cv2
import numpy as np
input_path = 'E:/Palm-Extraction/test_images/1.jpg'
img_original = cv2.imread(input_path)
# h, w, _= img_original.shape
# img0 = np.zeros((h+160, w, 3), np.uint8)
# img0[80:-80,:] = img_original    
img = cv2.cvtColor(img_original, cv2.COLOR_BAYER_BG2GRAY)
cv2.imshow(img)