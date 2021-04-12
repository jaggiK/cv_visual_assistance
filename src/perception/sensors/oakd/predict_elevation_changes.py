import cv2
import glob
import numpy as np
import os

import time
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from pyimage_ml.pyimagesearch.datasets import SimpleDatasetLoader
from pyimage_ml.pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import sys
sys.path.append("../../../")
from perception.openvino.examples.edge_app import app as openvino_app
from perception.openvino.examples.edge_app import inference


IMG_WIDTH = 64
IMG_HEIGHT = 64

img_model = load_model("trained_model")
depth_model = load_model("depth_trained_model.model")

openvino_model_im = "/home/jaggi/elevation_model/image_based/saved_model.xml"
# Create a Network for using the Inference Engine
inference_network = inference.Network()
# Load the model in the network, and obtain its input shape
n, c, h, w = inference_network.load_model(openvino_model_im, "HETERO:CPU,MYRIAD", "DUMMY")


openvino_model_depth = "/home/jaggi/elevation_model/depth_based/saved_model.xml"
# Create a Network for using the Inference Engine
depth_network = inference.Network()
# Load the model in the network, and obtain its input shape
n, c, h, w = depth_network.load_model(openvino_model_depth, "HETERO:CPU,MYRIAD", "DUMMY")

def preprocess_image_for_elevation(image):
    image = cv2.resize(image, (640, 360))
    crop_img = image[250:350, 200:375]
    crop_img = cv2.resize(crop_img, (IMG_WIDTH, IMG_HEIGHT))
    crop_img = crop_img.astype("float") / 255.0
    return crop_img


def preprocess_depth_image_for_elevation(depth_img):
    depth_img = depth_img / 1000.0
    depth_img = cv2.resize(depth_img, (IMG_WIDTH, IMG_HEIGHT))
    depth_img = depth_img.reshape((IMG_WIDTH, IMG_HEIGHT, 1))
    return depth_img


for f in glob.glob("/home/jaggi/depth_image_dataset/down/test/*.jpg"):
    data_x = []
    print(f)
    image = cv2.imread(f)
    depth_img = np.load(f.replace(".jpg", ".npz"))["cropped"]
    start = time.time()
    elev_im_output = openvino_app.perform_inference_elev_img(image, inference_network,
                                                   h, w)
    # print(type(elev_im_output))
    elev_depth_output = openvino_app.perform_inference_elev_depth(depth_img, depth_network,
                                                             h, w)
    pred_val_img = elev_im_output[np.argmax(elev_im_output)]
    pred_val_depth = elev_depth_output[np.argmax(elev_im_output)]
    end = time.time()
    print("secs = ", end-start)
    print("depth open vino=", elev_depth_output)
    print("img open vino=", elev_im_output, pred_val_depth + pred_val_img)
    process_img = preprocess_image_for_elevation(image)
    depth_process_img = preprocess_depth_image_for_elevation(depth_img)

    process_img = process_img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))
    depth_process_img = depth_process_img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 1))

    start = time.time()
    img_prediction = np.argmax(img_model.predict(process_img, batch_size=1)[0])
    depth_prediction = np.argmax(depth_model.predict(depth_process_img, batch_size=1)[0])
    end = time.time()
    # print("time taken = ", end - start)
    # print("img preds = ", img_prediction)
    # print("depth_preds = ", depth_prediction)
