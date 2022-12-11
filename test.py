import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('model/keras_model.h5', 'model/labels.txt')
labels = ['A', 'B', 'C']
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    try:
        if hands:
            hand = hands[0]
            # getting bounding box information
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3),
                               np.uint8) * 255  # here without * 255 get a black window because only selecting ones

            # crop image -- starting [height: ending height][y:y+1], starting [width:ending width][x:x+1]
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # this will give the exact bounding box

            # imgCrop = imgCrop.shape

            aspectRatio = h / w  # if value is above one h is grater is value is below one w is grater

            if aspectRatio > 1:
                k = imgSize / h  # stretching the height
                wCal = math.ceil(k * w)  # mul k with the previous width and also adding math.ceil (if wCal= 3.5 => 4)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                # set the image at center on imgWhite
                wGap = math.ceil((imgSize - wCal) / 2)
                # adding detected hand on top of imgWhite
                imgWhite[:, wGap:wCal+wGap] = imgResize
                # adding the classifier
                prediction, index = classifier.getPrediction(img)
                # print(prediction, index)
            else:
                k = imgSize / w  # stretching the height
                hCal = math.ceil(k * h)  # mul k with the previous width and also adding math.ceil (if wCal= 3.5 => 4)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                # set the image at center on imgWhite
                hGap = math.ceil((imgSize - hCal) / 2)
                # adding detected hand on top of imgWhite
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImgeWhite', imgWhite)
    except:
        pass
    cv2.imshow('Image', img)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyWindow()
