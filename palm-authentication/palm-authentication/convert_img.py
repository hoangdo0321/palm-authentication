import cv2
import numpy as np
class Convert:
    def __init__(self, img_path):
        self.img = cv2.resize(cv2.imread(img_path)[150:340, 380: 580], (600, 600))

    def convert_img(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        noise = cv2.fastNlMeansDenoising(gray)
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        print ("reduced noise")

        # equalist hist
        kernel = np.ones((7,7),np.uint8)
        self.img = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
        img_yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        print ("equalized hist")

        # invert
        inv = cv2.bitwise_not(img_output)
        print ("inverted")

        # erode
        gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
        erosion = cv2.erode(gray,kernel,iterations = 1)
        print ("eroded")

        # skel
        self.img = gray.copy()
        skel = self.img.copy()
        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        iterations = 0

        while True:
            eroded = cv2.morphologyEx(self.img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(self.img, temp)
            skel = cv2.bitwise_or(skel, temp)
            self.img[:,:] = eroded[:,:]
            if cv2.countNonZero(self.img) == 0:
                break

        print ("skeletonized")
        ret, thr = cv2.threshold(skel, 5,255, cv2.THRESH_BINARY)
        return thr
            
# conv = Convert('E:/Palm-Authentication/test_pic/pic 3.jpg')
# img = conv.convert()
# cv2.imwrite('sample1.jpg',img)
