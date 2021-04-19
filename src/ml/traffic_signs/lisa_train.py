# USAGE
# python lisa_train.py --dataset lisa_dataset/signDatabasePublicFramesOnly/cropped_images --model output/lisa_trafficsignnet.model --plot output/lisa_plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import tensorflow as tf
from pyimagesearch.trafficsignnet import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import json
from config import lisa_config_for_create_images as config
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to training history plot")
args = vars(ap.parse_args())

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_images_and_labels():
	# grab the list of images that we'll be describing
	print("[INFO] loading images...")
	imagePaths = list(paths.list_images(args["dataset"]))

	# initialize the image preprocessor, load the dataset from disk,
	# and reshape the data matrix
	sp = SimplePreprocessor(32, 32)
	sdl = SimpleDatasetLoader(preprocessors=[sp])
	(images, label) = sdl.load(imagePaths, verbose=500)
	#image = image.reshape((image.shape[0], 3072))
	# resize the image to be 32x32 pixels, ignoring aspect ratio,
	# and then perform Contrast Limited Adaptive Histogram
	# Equalization (CLAHE)
	resized_images = []
	for image in images:
		#print("image shape = ", image.shape[1::-1])
		r_image = transform.resize(image, (32, 32))
		r_image = exposure.equalize_adapthist(r_image, clip_limit=0.1)
		resized_images.append(r_image)

	# show some information on memory consumption of the images
	#print("[INFO] features matrix: {:.1f}MB".format(resized_images.nbytes / (1024 * 1024.0)))

	# convert the data and labels to NumPy arrays
	resized_images = np.array(resized_images)
	label = np.array(label)

	return resized_images, label

# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

# load the label names
#labelNames = open("signnames.csv").read().strip().split("\n")[1:]
#labelNames = [l.split(",")[1] for l in labelNames]

labelNames = list(config.CLASSES)

data, labels = get_images_and_labels()

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

# account for skew in the labeled data
classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3,
	classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# compile the model and train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])
print("network model saved: ", args["model"])


# plot the training loss and accuracy
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
print("plot saved: ", args["plot"])

print("training is complete")
