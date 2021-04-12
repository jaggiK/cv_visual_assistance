import cv2
import glob
import numpy as np
import os

import time



IMG_WIDTH = 64
IMG_HEIGHT = 64


def preprocess(image):
    image = cv2.resize(image, (640, 360))
    crop_img = image[250:350, 200:375]
    crop_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))
    crop_img = crop_img.astype("float") / 255.0
    return crop_img


data_x = []
data_y = []
for f in glob.glob("/home/jaggi/depth_image_dataset/up/test/*.jpg"):
    print(f)
    image = cv2.imread(f)
    process_img = preprocess(image)
    data_x.append(process_img)
    data_y.append(1)

for f in glob.glob("/home/jaggi/depth_image_dataset/down/test/*.jpg"):
    print(f)
    image = cv2.imread(f)
    process_img = preprocess(image)
    data_x.append(process_img)
    data_y.append(2)

for f in glob.glob("/home/jaggi/depth_image_dataset/flat/test/*.jpg"):
    print(f)
    image = cv2.imread(f)
    process_img = preprocess(image)
    data_x.append(process_img)
    data_y.append(0)
data_x = np.array(data_x)
data_y = np.array(data_y)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from pyimage_ml.pyimagesearch.datasets import SimpleDatasetLoader
from pyimage_ml.pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint


(trainX, testX, trainY, testY) = train_test_split(data_x, data_y,
test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3,
classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])



checkpoint = ModelCheckpoint("trained_model", monitor="val_loss",
save_best_only=True, verbose=1)
callbacks = [checkpoint]
# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1, callbacks=callbacks)
