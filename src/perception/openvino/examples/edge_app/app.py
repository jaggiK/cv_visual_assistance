import sys
import argparse
import time
import threading

import cv2
import numpy as np
import glob
from operator import itemgetter
from skimage import io
import imutils

sys.path.append("../../../../")
from perception.openvino.examples.edge_app.handle_models import handle_output, preprocessing, \
    preprocess_sign_classifier, \
    preprocess_elev_img, preprocess_depth_image
from perception.openvino.examples.edge_app.inference import Network

# initialize the class labels dictionary
TRAFFIC_SIGN_CLASSES = ["pedestrianCrossing", "signalAhead", "stop"]
TRAFFIC_SIGN_CLASSES = ["pedestrian_sign", "sidewalk_closed", "signal_ahead", "slow", "stop_sign", "stop_ahead"]

CAR_COLORS = ["white", "gray", "yellow", "red", "green", "blue", "black"]
CAR_TYPES = ["car", "bus", "truck", "van"]
SEM_SEG_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
                   "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                   "bicycle", "ego-vehicle"]
ROAD_SEG_CLASSES = ['BG', 'road', 'curb', 'mark']
TRAFFIC_DETECTOR_LABELS = ["background",
                           "fire_hydrant",
                           "handicap_sign",
                           "push_button",
                           "public_trash_can",
                           "street_name",
                           "handicap_sign",
                           "traffic_cone",
                           "traffic_light",
                           "text-board",
                           "traffic_sign",
                           "trash_logo",
                           "yellow_pavement"

                           ]

SEM_SEG_COLORS = [[0, 0, 255], [0, 255, 0], [128, 128, 0], [100, 255, 200], [212, 255, 127], [255, 0, 255],
                  [100, 0, 100],
                  [0, 100, 100], [0, 255, 255], [100, 100, 100], [19, 69, 139], [255, 0, 125], [200, 90, 175],
                  [255, 0, 0], [200, 200, 200], [50, 50, 50], [20, 20, 20], [80, 20, 200], [10, 100, 200],
                  [200, 200, 100]]
ROAD_SEG_COLORS = [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]]

ROAD_SEG_IMAGE_SIZE = [512, 896]
SEG_IMAGE_SIZE = [1024, 2048]
# create image for legend
SEM_LEGEND = np.zeros((SEG_IMAGE_SIZE[0], 180, 3))
j = 0
legend_height = int(SEG_IMAGE_SIZE[0] / len(SEM_SEG_CLASSES))
for i in range(0, len(SEM_SEG_CLASSES)):
    SEM_LEGEND[j:j + legend_height, :] = SEM_SEG_COLORS[i]
    legend_reg = SEM_LEGEND[j:j + legend_height, :]
    cv2.putText(legend_reg, SEM_SEG_CLASSES[i], (int(legend_reg.shape[0] / 2), int(legend_reg.shape[1] / 2) - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    j = j + legend_height

PERSON_ATTR = ["is_male", "has_bag", "has_backpack", "has_hat", "has_longsleeves", "has_longpants",
               "has_longhair", "has_coat_jacket"]


def get_args():
    """
    Gets the arguments from the command line.
    """

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"
    t_desc = "The type of model: POSE, TEXT or CAR_META"
    f_desc = "Folder containing images"
    v_desc = "Display output image"
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=False)
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-t", help=t_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    optional.add_argument("-f", help=f_desc, default=None)
    optional.add_argument("-v", help=v_desc, default=False)
    args = parser.parse_args()

    return args


def get_mask(processed_output):
    """
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    """
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def cross_walk_detection(seg_image):
    seg_binary_image = np.zeros((seg_image.shape[0], seg_image.shape[1]))
    mark_indices = np.where(np.all(seg_image == (255, 0, 0), axis=-1))
    seg_binary_image[mark_indices] = 255
    # find all your connected components (white blobs in your image)
    seg_binary_image = seg_binary_image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(seg_binary_image, connectivity=8)
    print("number of comps = ", nb_components)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    min_size = 450
    img2 = np.zeros((output.shape))
    print(img2.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    img2 = cv2.dilate(img2, None, iterations=8)
    img2 = img2.astype('uint8')
    # img2[600:, :] = 0
    cnts = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        area = cv2.contourArea(c)
        print(area)
        if area > 5:
            box = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))

            cv2.drawContours(seg_image, [box], -1, (0, 255, 255), 2)
            (x, y), (MA, ma), angle = cv2.fitEllipse(c)
            hull = cv2.convexHull(c)
            hullArea = cv2.contourArea(hull)
            solidity = area / float(hullArea)
    # cv2.imshow("road_marks", seg_binary_image)
    # cv2.imshow("processed", img2)
    # cv2.imshow("processed2", seg_image)
    return seg_image


def create_output_image(model_type, image, output):
    """
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    """
    if model_type == "POSE":
        # Remove final part of output not used for heatmaps
        output = output[:-1]
        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c] > 0.5, 255, 0)
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        image = image + pose_mask
    elif model_type == "TEXT":
        # Get only text detections above 0.5 confidence, set to 255
        output = np.where(output[1] > 0.5, 255, 0)
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = image + text_mask
    elif model_type == "CAR_META":
        # Get the color and car type from their lists
        color = CAR_COLORS[output[0]]
        car_type = CAR_TYPES[output[1]]
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        image = cv2.putText(image,
                            "Color: {}, Type: {}".format(color, car_type),
                            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
                            2 * scaler, (255, 255, 255), 3 * scaler)
    elif model_type == "PER" or model_type == "VEH" or model_type == "VEH_BIKE" or model_type == "TDET":
        # discard detections <= 0.5
        valid_detections = output[output[:, 2] > 0.5]
        for detection in valid_detections:
            ymin = int(image.shape[0] * detection[4])
            xmin = int(image.shape[1] * detection[3])
            ymax = int(image.shape[0] * detection[6])
            xmax = int(image.shape[1] * detection[5])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # print(TRAFFIC_DETECTOR_LABELS[int(detection[1])])
            if model_type == "TDET":
                if not TRAFFIC_DETECTOR_LABELS[int(detection[1])] == "traffic_sign":
                    cv2.putText(image, TRAFFIC_DETECTOR_LABELS[int(detection[1])], (xmin, ymin),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0))
    elif model_type == "SEG":
        image = cv2.resize(image, (output.shape[1], output.shape[0]))
        orig_image = image.copy()
        for c in range(0, len(SEM_SEG_CLASSES)):
            image[np.where(output == c)] = SEM_SEG_COLORS[c]
        image = cv2.addWeighted(orig_image, 0.3, image, 0.7, 0)
        image = np.hstack((image, SEM_LEGEND.astype('uint8')))
    elif model_type == "PER_ATTR":
        attrs = output[0]
        top_point = output[1]
        bottom_point = output[2]
        pred_classes = list(np.where(attrs > 0.5)[0].flatten())
        if len(pred_classes):
            class_str = list(itemgetter(*pred_classes)(PERSON_ATTR))
            if pred_classes[0] != 0:
                class_str.insert(0, "is_female")
        else:
            class_str = ["is_female"]
        print(class_str)
        # TODO (jagadish) : below logic to be verified
        topy = int(top_point[1] * image.shape[0])
        topx = int(top_point[0] * image.shape[1])
        bottomy = int(bottom_point[1] * image.shape[0])
        bottomx = int(bottom_point[0] * image.shape[1])

        cv2.circle(image, (topx, topy), 2, (0, 0, 255), 2)
        cv2.circle(image, (bottomx, bottomy), 2, (0, 0, 255), 2)
    elif model_type == "ROAD_SEG":
        preds = np.argmax(output, axis=0)
        image = cv2.resize(image, (output.shape[2], output.shape[1]))
        orig_image = image.copy()
        for c in range(0, len(ROAD_SEG_CLASSES)):
            image[np.where(preds == c)] = ROAD_SEG_COLORS[c]
        # image = np.hstack((image, SEM_LEGEND.astype('uint8')))
    elif model_type == "TSIGN":
        traff_sign_class = np.argmax(output)
        print("detected traffic sign = ", TRAFFIC_SIGN_CLASSES[traff_sign_class])
        image = imutils.resize(image, width=128)
        cv2.putText(image, TRAFFIC_SIGN_CLASSES[traff_sign_class], (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)

    return image


def perform_inference(args, inference_network, h, w, request_id, request_id_list=None, async_flag=False):
    """
    Performs inference on an input image, given a model.
    """

    # Read the input image
    image = cv2.imread(args.i)

    # Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    start_time = time.time()

    if async_flag:
        ### Perform inference on the frame
        inference_network.async_inference(preprocessed_image, request_id)

        ### Get the output of inference
        while True:
            status = inference_network.wait(request_id)
            if status == 0:
                break

        output = inference_network.extract_output_async(request_id)
        if request_id in request_id_list:
            request_id_list.remove(request_id)

    else:
        # Obtain the output of the inference request
        # Perform synchronous inference on the image
        inference_network.sync_inference(preprocessed_image)

        output = inference_network.extract_output()

    # Handle the output of the network, based on args.t
    # This will require using `handle_output` to get the correct
    # function, and then feeding the output to that function.
    output_func = handle_output(args.t)
    processed_output = output_func(output, image.shape)

    end_time = time.time()
    print(f"inference time for one frame : {end_time - start_time} secs")
    # Create an output image based on network
    output_image = create_output_image(args.t, image, processed_output)
    # output_image = cross_walk_detection(output_image)
    # Save down the resulting image
    # outfile_path = "outputs/{}-output_INT8.png".format(args.i.split("/")[-1].split(".")[0])
    # print('outfile_path = ', outfile_path)
    # cv2.imwrite(outfile_path, output_image)

    if args.v:
        cv2.imshow("image", output_image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def perform_inference_seg(img, inference_network, h, w):
    """
    Performs inference on an input image, given a model.
    """

    # Read the input image
    image = img

    # Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    start_time = time.time()
    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)
    # Obtain the output of the inference request
    output = inference_network.extract_output()

    # Handle the output of the network, based on args.t
    # This will require using `handle_output` to get the correct
    # function, and then feeding the output to that function.
    output_func = handle_output("SEG")
    processed_output = output_func(output, image.shape)

    end_time = time.time()
    print(f"seg inference time for one frame : {end_time - start_time} secs")
    # Create an output image based on network
    output_image = create_output_image("SEG", image, processed_output)

    if True:
        cv2.imshow("image", cv2.resize(output_image, (700, 350)))


def perform_inference_traffic_signs(img, inference_network, h, w):
    """
    Performs inference on an input image, given a model.
    """

    # Read the input image
    image = img
    # Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    start_time = time.time()
    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)
    # Obtain the output of the inference request
    output = inference_network.extract_output()

    # Handle the output of the network, based on args.t
    # This will require using `handle_output` to get the correct
    # function, and then feeding the output to that function.
    output_func = handle_output("TDET")
    processed_output = output_func(output, image.shape)

    end_time = time.time()
    print(f"inference time for one frame : {end_time - start_time} secs")
    # Create an output image based on network
    output_image = create_output_image("TDET", image, processed_output)

    if True:
        # cv2.imshow("tf_image", cv2.resize(output_image, (700, 350)))
        pass
    return processed_output

timestr = time.strftime("%Y%m%d-%H%M%S")
seg_video = cv2.VideoWriter('videos_rec/seg_video_{}.avi'.format(timestr),
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            3, (896, 512))


def perform_inference_road_seg(img, inference_network, h, w, crosswalk=False):
    """
    Performs inference on an input image, given a model.
    """

    # Read the input image
    image = img

    # Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    start_time = time.time()
    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)
    # Obtain the output of the inference request
    output = inference_network.extract_output()

    # Handle the output of the network, based on args.t
    # This will require using `handle_output` to get the correct
    # function, and then feeding the output to that function.
    output_func = handle_output("ROAD_SEG")
    processed_output = output_func(output, image.shape)

    end_time = time.time()
    print(f"road seg inference time for one frame : {end_time - start_time} secs")
    # Create an output image based on network
    output_image = create_output_image("ROAD_SEG", image, processed_output)
    if crosswalk:
        output_image = cross_walk_detection(output_image)
    orig_image = image.copy()
    orig_image = cv2.resize(orig_image, (output_image.shape[1], output_image.shape[0]))
    # print("--------------------------------------------------------- seg image shape ", output_image.shape)
    if True:
        cv2.imshow("road_segmentation", cv2.addWeighted(orig_image, 0.5, output_image, 0.5, 0))
        seg_video.write(cv2.addWeighted(orig_image, 0.5, output_image, 0.5, 0))


def perform_inference_signs(args, inference_network, h, w):
    image = io.imread(args.i)
    preprocessed_image = preprocess_sign_classifier(image)
    inference_network.sync_inference(preprocessed_image)
    output = inference_network.extract_output()
    output_func = handle_output("TSIGN")
    processed_output = output_func(output, image.shape)
    output_image = create_output_image("TSIGN", image, processed_output)
    # Save down the resulting image
    outfile_path = "outputs/{}.png".format(args.i.split("/")[-1].split(".")[0])
    print('outfile_path = ', outfile_path)
    cv2.imwrite(outfile_path, output_image)


def perform_inference_signs_img(image, inference_network, h, w):
    preprocessed_image = preprocess_sign_classifier(image)
    inference_network.sync_inference(preprocessed_image)
    output = inference_network.extract_output()
    output_func = handle_output("TSIGN")
    processed_output = output_func(output, image.shape)
    traff_sign_class = np.argmax(processed_output)

    return traff_sign_class


def perform_inference_elev_img(image, inference_network, h, w):
    preprocessed_image = preprocess_elev_img(image)
    inference_network.sync_inference(preprocessed_image)
    output = inference_network.extract_output()
    output_func = handle_output("ELE_IM")
    processed_output = output_func(output, image.shape)
    # elev_img_class = np.argmax(processed_output)

    return processed_output


def perform_inference_elev_depth(depth_image, inference_network, h, w):
    preprocessed_image = preprocess_depth_image(depth_image)
    inference_network.sync_inference(preprocessed_image)
    output = inference_network.extract_output()
    output_func = handle_output("ELE_DEPTH")
    processed_output = output_func(output, depth_image.shape)
    # elev_img_class = np.argmax(processed_output)

    return processed_output


def main():
    args = get_args()
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)
    print(f"n: {n}, c:{c}, h:{h}, w ={w}")
    request_id = 0
    request_id_lists = []
    threads = []
    if args.f:
        start_time = time.time()
        for fname in glob.glob(args.f + "/*.jpg"):
            args.i = fname
            # perform_inference(args, inference_network, h, w, request_id)
            perform_inference(args, inference_network, h, w, 1)
        end_time = time.time()
        print("time taken : ", (end_time - start_time))
    else:
        perform_inference(args, inference_network, h, w, request_id)


def async_main():
    args = get_args()
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)
    request_id = 0
    request_id_free = []
    request_ids_occupied = []
    threads = []
    if args.f:
        for i in range(inference_network.num_requests):
            request_id_free.append(i)

        start_time = time.time()
        for fname in glob.glob(args.f + "/*.jpg"):
            args.i = fname

            # Read the input image
            image = cv2.imread(args.i)

            # Preprocess the input image
            preprocessed_image = preprocessing(image, h, w)

            ### Perform inference on the frame
            print(time.time())
            if len(request_id_free) > 0:
                start_time_frame = time.time()
                request_id = request_id_free.pop(0)
                print(f"request id {request_id} started at {start_time_frame}")
                inference_network.async_inference(preprocessed_image, request_id)
                request_ids_occupied.append(request_id)

            for req_id in request_ids_occupied:
                status = inference_network.wait(req_id)
                if status == 0:
                    request_id_free.append(req_id)
                    request_ids_occupied.remove(req_id)
                    output = inference_network.extract_output_async(request_id)
                    output_func = handle_output(args.t)
                    processed_output = output_func(output, image.shape)
                    end_time_frame = time.time()
                    print(
                        f"inference time for one frame, req id {req_id}: {end_time_frame - start_time_frame} secs, ended at {end_time_frame}")
                else:
                    print("not processed, will revisit")

        end_time = time.time()
        print("time taken : ", (end_time - start_time))


if __name__ == "__main__":
    # async_main()
    main()
