# from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12

# model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

# model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

# model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

# out = model.predict_segmentation(
#     inp="/home/jaggi/oakd_dataset/dataset_for_labelling/resized/862_resized.jpg",
#     out_fname="out.png"
# )

# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import cv2


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    print("num_labels = ", numLabels)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

"""
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("/home/jaggi/oakd_dataset/dataset_for_labelling/resized/862_resized.jpg")
# image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# image = image[int(image.shape[0]/4) :, :]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)
# Detect points that form a line
lines = cv2.HoughLinesP(auto, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=10)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imshow("Result Image", image)
# show the images
cv2.imshow("Original", image)
cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.waitKey(0)
"""

cropped_frame = np.load("/home/jaggi/depth_image_dataset/mat14.npz")["cropped"]
full_frame = np.load("/home/jaggi/depth_image_dataset/mat14.npz")["full"]
cropped_frame = cv2.applyColorMap(cropped_frame, cv2.COLORMAP_HOT)
full_frame = cv2.applyColorMap(full_frame, cv2.COLORMAP_HOT)
cv2.imshow("cropped_frame", cropped_frame)
cv2.imshow("full_frame", full_frame)
cv2.waitKey(0)

img = cv2.imread("/home/jaggi/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/2007_000392.png")
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
