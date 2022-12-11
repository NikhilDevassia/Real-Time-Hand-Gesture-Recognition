import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    try:
        if hands:
            hand = hands[0]
            # getting bounding box information
            x, y, w, h = hand['bbox']
            # crop image -- starting [height: ending height][y:y+1], starting [width:ending width][x:x+1]
            imgCrop = img[y - offset:y + offset + h, x - offset:x + offset + w]  # this will give the exact bounding box
            cv2.imshow('ImageCrop', imgCrop)
    except:
        pass
    cv2.imshow('Image', img)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyWindow()

