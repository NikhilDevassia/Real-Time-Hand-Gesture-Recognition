import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = 'Data/C'
counter = 0

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
    if k == ord('s'):
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite) # time.time() will give different values
        counter += 1
        print(counter)
    if k == 27:
        cv2.destroyWindow()
