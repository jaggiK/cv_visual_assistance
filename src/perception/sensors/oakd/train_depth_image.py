import cv2
import glob
import numpy as np
import os

import time



IMG_WIDTH = 64
IMG_HEIGHT = 64





data_x = []
data_y = []
for f in glob.glob("/home/jaggi/depth_image_dataset/up/test/*.npz"):
    print(f)
    depth_f = np.load(f)["cropped"]
    depth_f = depth_f/1000.0
    depth_f = cv2.resize(depth_f, (IMG_WIDTH, IMG_HEIGHT))
    depth_f = depth_f.reshape((IMG_WIDTH, IMG_HEIGHT, 1))
    data_x.append(depth_f)
    data_y.append(1)

for f in glob.glob("/home/jaggi/depth_image_dataset/down/test/*.npz"):
    print(f)
    depth_f = np.load(f)["cropped"]
    depth_f = depth_f / 1000.0
    depth_f = cv2.resize(depth_f, (IMG_WIDTH, IMG_HEIGHT))
    depth_f = depth_f.reshape((IMG_WIDTH, IMG_HEIGHT, 1))
    data_x.append(depth_f)
    data_y.append(2)

for f in glob.glob("/home/jaggi/depth_image_dataset/flat/test/*.npz"):
    print(f)
    depth_f = np.load(f)["cropped"]
    depth_f = depth_f / 1000.0
    depth_f = cv2.resize(depth_f, (IMG_WIDTH, IMG_HEIGHT))
    depth_f = depth_f.reshape((IMG_WIDTH, IMG_HEIGHT, 1))
    data_x.append(depth_f)
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
model = MiniVGGNet.build(width=64, height=64, depth=1,
classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])



checkpoint = ModelCheckpoint("depth_trained_model.model", monitor="val_loss",
save_best_only=True, verbose=1)
callbacks = [checkpoint]
# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=100, verbose=1, callbacks=callbacks)
