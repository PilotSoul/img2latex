from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from imutils import paths
import pandas as pd
import numpy as np
import random
import pickle
import cv2
import os
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def segment_image():
    img = cv2.imread('numbers.jpg')
    custom_config = r'--oem 3 --psm 6'
    hImg, wImg, _ = img.shape
    boxes = pytesseract.image_to_boxes(img, config=custom_config)
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), hImg - int(b[2])), (int(b[3]), hImg- int(b[4])), (0, 255, 0), 1)
        print(b)

    cv2.imshow('Detected text', img)
    cv2.waitKey(0)