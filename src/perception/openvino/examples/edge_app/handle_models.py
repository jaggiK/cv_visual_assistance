import cv2
import numpy as np
from skimage import transform
from skimage import exposure

"""
Note : Lot of logic in this file is tied with model architecture
       check out the OpenVINO model documents
"""


def handle_pose(output, input_shape):
    """
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    """
    # Extract only the second blob output (keypoint heatmaps)
    heatmaps = output['Mconv7_stage2_L2']
    # Resize the heatmap back to the size of the input
    # Create an empty array to handle the output map
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    # Iterate through and re-size each heatmap
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap


def handle_text(output, input_shape):
    """
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    """
    # Extract only the first blob output (text/no text classification)
    text_classes = output['model/segm_logits/add']
    # Resize this output back to the size of the input
    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])

    return out_text


def handle_car(output, input_shape):
    """
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    """
    # Get rid of unnecessary dimensions
    color = output['color'].flatten()
    car_type = output['type'].flatten()
    # Get the argmax of the "color" output
    color_pred = np.argmax(color)
    # Get the argmax of the "type" output
    type_pred = np.argmax(car_type)

    return color_pred, type_pred


def handle_object_detection(output, input_shape):
    """
    model output shape is [1X1XNX7]
    N -> number of detections
    7 -> image_id, label, conf, xmin, ymin, xmax, ymax
    """
    detections = output['detection_out'][0][0]
    return detections


def handle_semantic(output, input_shape):
    segmented_image = output['4119.1'][0][0]
    return segmented_image


def handle_person_attributes(output, input_shape):
    person_attrs = output['453'].flatten()
    top_point = output['456'][0].flatten()
    bottom_point = output['459'][0].flatten()
    return person_attrs, top_point, bottom_point


def handle_vehicle_detection(output, input_shape):
    detections = output['detection_out'][0][0]
    print(detections)
    return detections


def handle_road_segmentation(output, input_shape):
    detections = output['L0317_ReWeight_SoftMax'][0]
    return detections


def handle_traffic_sign_classifier(output, input_shape):
    # print(output['StatefulPartitionedCall/sequential/activation_7/Softmax'][0])
    return output['StatefulPartitionedCall/sequential/activation_7/Softmax'][0]


def handle_elev_im_classifier(output, input_shape):
    # print("pred = ", output['StatefulPartitionedCall/sequential/activation_5/Softmax'][0])
    return output['StatefulPartitionedCall/sequential/activation_5/Softmax'][0]

def handle_elev_depth_classifier(output, input_shape):
    # print(output)
    return output['StatefulPartitionedCall/sequential/activation_5/Softmax'][0]

def handle_traffic_object_detection(output, input_shape):
    """
    model output shape is [1X1XNX7]
    N -> number of detections
    7 -> image_id, label, conf, xmin, ymin, xmax, ymax
    """
    detections = output['DetectionOutput'][0][0]
    return detections


def handle_output(model_type):
    """
    Returns the related function to handle an output,
        based on the model_type being used.
    """
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    elif model_type == "PER" or model_type == "VEH" or model_type == "VEH_BIKE":
        return handle_object_detection
    elif model_type == "SEG":
        return handle_semantic
    elif model_type == "PER_ATTR":
        return handle_person_attributes
    elif model_type == "ROAD_SEG":
        return handle_road_segmentation
    elif model_type == "TSIGN":
        return handle_traffic_sign_classifier
    elif model_type == "TDET":
        return handle_traffic_object_detection
    elif model_type == "ELE_IM":
        return handle_elev_im_classifier
    elif model_type == "ELE_DEPTH":
        return handle_elev_depth_classifier
    else:
        return None


"""
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
"""


def preprocessing(input_image, height, width):
    """
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    """
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    # depth, height, width
    image = image.transpose((2, 0, 1))
    # add batch size
    image = image.reshape(1, 3, height, width)

    return image


def preprocess_sign_classifier(image):
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)

    # preprocess the image by scaling it to the range [0, 1]
    # image = image.astype("float32") / 255.0
    # image = np.expand_dims(image, axis=0)
    image = image.transpose((2, 0, 1))
    # add batch size
    image = image.reshape(1, 3, 32, 32)
    return image


IMG_WIDTH = 64
IMG_HEIGHT = 64


def preprocess_elev_img(image):
    image = cv2.resize(image, (640, 360))
    crop_img = image[250:350, 200:375]
    crop_img = cv2.resize(crop_img, (64, 64))
    crop_img = crop_img.astype("float") / 255.0
    crop_img = crop_img.transpose((2, 0, 1))
    crop_img = crop_img.reshape(1, 3, 64, 64)
    return crop_img


def preprocess_depth_image(depth_img):
    depth_img = depth_img / 1000.0
    depth_img = cv2.resize(depth_img, (IMG_WIDTH, IMG_HEIGHT))
    # depth_img = depth_img.reshape((IMG_WIDTH, IMG_HEIGHT, 1))
    # depth_img = depth_img.transpose((2,0,1))
    depth_img = depth_img.reshape((IMG_WIDTH, IMG_HEIGHT))
    return depth_img
