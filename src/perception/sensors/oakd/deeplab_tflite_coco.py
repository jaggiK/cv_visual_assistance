# -*- coding: utf-8 -*-
"""DeepLab_TFLite_COCO.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/DeepLabV3/DeepLab_TFLite_COCO.ipynb

Reference: https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb. Thanks to [Khanh](https://twitter.com/khanhlvg) & [Meghna](https://twitter.com/natrajmeghna) for their help and guidance. The models used here were trained on the Pascal VOC 2012 dataset.

## Setup
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
import time

print(tf.__version__)

dynamic_tflite_path = "/home/jaggi/Downloads/f32.tflite"
f16_tflite_path = "/home/jaggi/Downloads/pascal_f16.tflite"
f16_tflite_path = "/home/jaggi/Downloads/cityscapes_f16.tflite"
f16_tflite_path = "/home/jaggi/Downloads/ade20k.tflite"
int_tflite_path = "/home/jaggi/Downloads/pascal_voc_int8.tflite"
"""## Inference using TFLite model

### 1. Get Input Image Size
"""

# @title Choose TFLite model type

model_dict = {
    "dynamic-range": dynamic_tflite_path,
    "fp16": f16_tflite_path,
    "int8": int_tflite_path
}

tflite_model_type = "fp16"  # @param ['dynamic-range', 'fp16', 'int8']

# Load the model.
interpreter = tf.lite.Interpreter(model_path=model_dict[tflite_model_type])

# Set model input.
input_details = interpreter.get_input_details()
interpreter.allocate_tensors()

# Get image size - converting from BHWC to WH
input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
print(input_size)

# @title 2. Provide a URL to your image to download
img_path = "/home/jaggi/Downloads/zuckerberg.png"
img_path = "/home/jaggi/oakd_dataset/test/706_lower_cam.jpg"
from PIL import Image
from PIL import ImageOps

image = Image.open(img_path)
image


def preprocess_for_tflite(image, input_size):

    """#### Prepare the downloaded image for running inference"""
    print(image)
    old_size = image.size  # old_size is in (width, height) format
    desired_ratio = input_size[0] / input_size[1]
    old_ratio = old_size[0] / old_size[1]

    if old_ratio < desired_ratio:  # '<': cropping, '>': padding
        new_size = (old_size[0], int(old_size[0] / desired_ratio))
    else:
        new_size = (int(old_size[1] * desired_ratio), old_size[1])

    print(new_size, old_size)

    # Cropping the original image to the desired aspect ratio
    delta_w = new_size[0] - old_size[0]
    delta_h = new_size[1] - old_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    cropped_image = ImageOps.expand(image, padding)
    cropped_image

    # Resize the cropped image to the desired model size
    resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)

    # Convert to a NumPy array, add a batch dimension, and normalize the image.
    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)
    image_for_prediction = image_for_prediction / 127.5 - 1
    return image_for_prediction, cropped_image


image_for_prediction, cropped_image = preprocess_for_tflite(image, input_size)

"""Thanks to Khanh for helping to figure out the pre-processing and post-processing code.

### 3. Run Inference
"""

# Load the model.
interpreter = tf.lite.Interpreter(model_path=model_dict[tflite_model_type])

# Invoke the interpreter to run inference.
interpreter.allocate_tensors()

import time
def predict_tflite(interpreter, image_for_prediction):
    t1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    interpreter.invoke()
    t2 = time.time()
    print("invoke = ", t2-t1)

    # Retrieve the raw output map.
    raw_prediction = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])()
    t3 = time.time()
    print("pred time= ", t3 - t2)
    # Post-processing: convert raw output to segmentation output
    ## Method 1: argmax before resize - this is used in some frozen graph
    # seg_map = np.squeeze(np.argmax(raw_prediction, axis=3)).astype(np.int8)
    # seg_map = np.asarray(Image.fromarray(seg_map).resize(image.size, resample=Image.NEAREST))
    ## Method 2: resize then argmax - this is used in some other frozen graph and produce smoother output

    width, height = cropped_image.size
    print("before argmax = ", raw_prediction.shape)
    # seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
    seg_map = np.argmax(raw_prediction, axis=3)
    print("after argmax seg map shape =", seg_map.shape)
    #seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
    seg_map = np.reshape(seg_map, (seg_map.shape[1], seg_map.shape[2]))
    print("after squeeze = ", seg_map.shape)
    t4 = time.time()
    print("argmax = ", t4 - t3)
    return seg_map

start_time = time.time()
seg_map = predict_tflite(interpreter, image_for_prediction)
end_time = time.time()

print("deeplab coco inference time = ", end_time-start_time)
"""The following code comes from https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb."""

# @title
from matplotlib import gridspec
from matplotlib import pyplot as plt


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


# def label_to_color_image(label):
#     """Adds color defined by the dataset colormap to the label.
#
#   Args:
#     label: A 2D array with integer type, storing the segmentation label.
#
#   Returns:
#     result: A 2D array with floating type. The element of the array
#       is the color indexed by the corresponding element in the input label
#       to the PASCAL color map.
#
#   Raises:
#     ValueError: If label is not of rank 2 or its value is larger than color
#       map maximum entry.
#   """
#     if label.ndim != 2:
#         raise ValueError('Expect 2-D input label')
#
#     colormap = create_pascal_label_colormap()
#
#     if np.max(label) >= len(colormap):
#         raise ValueError('label value too large.')
#
#     return colormap[label]


# def vis_segmentation(image, seg_map):
#     """Visualizes input image, segmentation map and overlay view."""
#     plt.figure(figsize=(15, 5))
#     grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
#
#     plt.subplot(grid_spec[0])
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title('input image')
#
#     plt.subplot(grid_spec[1])
#     seg_image = label_to_color_image(seg_map).astype(np.uint8)
#     plt.imshow(seg_image)
#     plt.axis('off')
#     plt.title('segmentation map')
#
#     plt.subplot(grid_spec[2])
#     plt.imshow(image)
#     plt.imshow(seg_image, alpha=0.7)
#     plt.axis('off')
#     plt.title('segmentation overlay')
#
#     unique_labels = np.unique(seg_map)
#     ax = plt.subplot(grid_spec[3])
#     plt.imshow(
#         FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
#     ax.yaxis.tick_right()
#     plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
#     plt.xticks([], [])
#     ax.tick_params(width=0.0)
#     plt.grid('off')
#     # plt.pause(0.2)
#     plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

LABEL_NAMES = np.asarray(["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
                   "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                   "bicycle", "ego-vehicle"])
import pandas as pd

ade20k_labels_info = pd.read_csv('objectInfo150.csv')
labels_list = list(ade20k_labels_info['Name'])
ade20k_labels_info.head()
labels_list.insert(0, 'others')
len(labels_list)

#@title
from matplotlib import gridspec
from matplotlib import pyplot as plt

def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_ade20k_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray(labels_list)

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
# FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
# FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

vis_segmentation(cropped_image, seg_map)

"""To try out a new model it's advisable to Factory Reset the runtime and then trying it."""