import sys
import time

import asyncio
import numpy as np
import cv2
import open3d as o3d

import copy
import subprocess
import argparse
import pickle

sys.path.append("../../../")
from oakd_config import config, decode_nn, show_nn, labels
from oakd import OAK_D
from perception.openvino.examples.edge_app import app as openvino_app
from perception.openvino.examples.edge_app import inference

from pointcloud_params import xy_coords, RELIABLE_DEPTH, DEPTH_THRESH, DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, \
    detect_left_right, intrinsics_right_cam

IMG_WIDTH = 64
IMG_HEIGHT = 64


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

def collect_depth(frame, frame_id, previewout=None):
    cropped_frame = frame[500:700, 400:750]
    np.savez('/home/jaggi/depth_image_dataset/mat{}.npz'.format(str(frame_id)), cropped=cropped_frame, full=frame)
    frame = (65535 // frame).astype(np.uint8).copy()
    cropped_frame = frame[500:700, 400:750]
    cropped_frame_display = cv2.applyColorMap(cropped_frame, cv2.COLORMAP_HOT)
    cv2.imshow("depth_cropped", cropped_frame_display)
    if previewout is not None:
        cv2.imwrite('/home/jaggi/depth_image_dataset/mat{}.jpg'.format(str(frame_id)), previewout)

timestr = time.strftime("%Y%m%d-%H%M%S")

depth_video = cv2.VideoWriter('videos_rec/depth_video_{}.avi'.format(timestr),
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         3, (1280, 720))

previewout_video = cv2.VideoWriter('videos_rec/prievewout_video_{}.avi'.format(timestr),
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              3, (640, 360))

def display_depth(frame, oakd, window_name, heights, widths, elev_flag, ostacle_grid=None):
    frame_copy = (65535 // frame).astype(np.uint8)
    frame_copy = cv2.applyColorMap(frame_copy, cv2.COLORMAP_HOT)
    cv2.rectangle(frame_copy, (400, 500), (750, 700), (255, 0, 0))
    cv2.rectangle(frame_copy, (200, 500), (950, 700), (255, 255, 0))
    if elev_flag:
        cv2.putText(frame_copy, 'X', (500,  560), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                cv2.LINE_AA)
    for row, (height, width) in enumerate(zip(heights, widths)):
        rect_width = width[1] - width[0]

        cv2.rectangle(frame_copy, (width[0], height[0]), (width[0] + int(rect_width * 1 / 4), height[1]), (0, 255, 255))
        if ostacle_grid[row][0] == True:
            cv2.putText(frame_copy, 'X', (width[0] + 50, height[0] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                        cv2.LINE_AA)

        cv2.rectangle(frame_copy, (width[0] + int(rect_width * 3 / 4), height[0]), (width[1], height[1]), (0, 255, 255))
        if ostacle_grid[row][1] == True:
            cv2.putText(frame_copy, 'X', (width[0] + int(rect_width * 3 / 4) + 50, height[0] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame_copy, (width[0] + int(rect_width * 1 / 4), height[0]),
                      (width[0] + int(rect_width * 3 / 4), height[1]), (0, 255, 255))
        if ostacle_grid[row][2] == True:
            cv2.putText(frame_copy, 'X', (width[0] + int(rect_width * 1 / 4) + 50, height[0] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("depth_cust", frame_copy)
    depth_video.write(frame_copy)
    cv2.putText(frame_copy, "depth", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
    cv2.putText(frame_copy, "fps: " + str(oakd.frame_count_prev[window_name]), (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 25)


def numpy_to_pcd(points, image=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if image is not None:
        pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
    return pcd




def get_args():
    """
    Gets the arguments from the command line.
    """

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "Child connection"
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-c", help=c_desc, default=None)
    args = parser.parse_args()

    return args




PROXIMITY_RANGE = 1.8


async def play_sound(audio_name):
    subprocess.Popen(["python3", "play_soundclip.py", "-f",
                      "{}".format(audio_name)])  # play_soundclip.py")#--filename {}".format(audio_name))

async def playaudio_name(name):
    subprocess.Popen(["python3", "playaudio_by_name.py", "-f",
                  "{}".format(name)])  # play_soundclip.py")#--filename {}".format(audio_name))


def update_user(voice_counter, obstacle_grid):
    obstacle_pkl = False
    if voice_counter > 20:
        audio_args = []
        for key, value in obstacle_grid.items():
            [obstacle_l, obstacle_r, obstacle_c] = value
            if obstacle_c or obstacle_r or obstacle_l:
                obstacle_pkl = True
            if key == 1:
                audio_args.extend(['t', int(obstacle_l), int(obstacle_r), int(obstacle_c)])
            elif key == 0:
                audio_args.extend(['b', int(obstacle_l), int(obstacle_r), int(obstacle_c)])
        if len(audio_args) > 0:
            loop = asyncio.get_event_loop()
            task2 = loop.create_task(play_sound(audio_args))
            loop.run_until_complete(asyncio.gather(task2))

        voice_counter = 0

        with open('obstacle_info.pkl', 'wb') as fh:
            pickle.dump(obstacle_pkl, fh)

    return voice_counter


def clock_angle(mid_x):
    if mid_x < 0.2:
        return "ten"
    if mid_x < 0.4:
        return "eleven"
    if mid_x < 0.6:
        return "twelve"
    if mid_x < 0.8:
        return "one"
    if mid_x < 1.0:
        return "two"
    else:
        return -1


unused_class = ["aeroplane", "bird", "boat", "cat", "cow", "horse"]
traffic_info = {"stop_sign": 0, "pedestrian_sign": 0, "signal_ahead": 0, "sidewalk_closed": 0,
                "last_t_stamp": 0, "stop_ahead":0, "slow" : 0}
traffic_time_info =  {"stop_sign": 0, "pedestrian_sign": 0, "signal_ahead": 0, "sidewalk_closed": 0,
                "last_t_stamp": 0, "stop_ahead":0, "slow" : 0, "crosswalk" : 0}

update_time_gap = 30
crosswalk_duration = 60

def flush_traffic_info(traffic_info, curr_timestamp):
    traffic_info["stop_sign"] = 0
    traffic_info["pedestrian_crossing"] = 0
    traffic_info["signal_ahead"] = 0
    traffic_info["sidewalk_closed"] = 0
    traffic_info["stop_ahead"] = 0
    traffic_info["slow"] = 0
    traffic_info["last_t_stamp"] = curr_timestamp
    return traffic_info

def update_time(traffic_time_info, sign, time):
    traffic_time_info[sign] = time


def main_2():
    # args = get_args()
    # print("---------", args.c)
    voice_counter = 0
    t_start = time.time()
    flush_traffic_info(traffic_info, t_start)
    jpeg_ctr = 0
    # get OAK-D instance - does init, creates pipeline + other initialization
    oakd = OAK_D(config)
    # model_path = "/home/jaggi/openvino_models/intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"
    model_path = "openvino_models/intel/road-segmentation-adas-0001/FP16-INT8/road-segmentation-adas-0001.xml"
    traffic_model_path = "openvino_models/intel/traffic_signs_model/FP16/traffic_detection_model.xml"
    tsign_classifier = "openvino_models/traffic_signs/saved_model.xml"
    openvino_model_elev_im = "openvino_models/elevation_model/image_based/saved_model.xml"
    inference_network = inference.Network()
    n, c, h, w = inference_network.load_model(model_path, "HETERO:CPU,MYRIAD,MYRIAD,MYRIAD", "DUMMY")

    # Create a Network for using the Inference Engine
    inference_network2 = inference.Network()
    # Load the model in the network, and obtain its input shape
    n_t, c_t, h_t, w_t = inference_network2.load_model(traffic_model_path, "HETERO:CPU,MYRIAD", "DUMMY")

    tsign_network2 = inference.Network()
    n_tc, c_tc, h_tc, w_tc = tsign_network2.load_model(tsign_classifier, "CPU", "DUMMY")

    # Create a Network for using the Inference Engine
    elev_im_network = inference.Network()
    n_e, c_e, h_e, w_e = elev_im_network.load_model(openvino_model_elev_im, "HETERO:CPU,MYRIAD", "DUMMY")

    openvino_model_depth = "openvino_models/elevation_model/depth_based/saved_model.xml"
    depth_network = inference.Network()
    n_d, c_d, h_d, w_d = depth_network.load_model(openvino_model_depth, "HETERO:CPU,MYRIAD", "DUMMY")

    if oakd.calibr_info is not None:
        print("right intrinsics : ", oakd.calibr_info["intrinsics_r"])
        print("right distortion : ", oakd.calibr_info["dist_r"])
        print("left intrinsics : ", oakd.calibr_info["intrinsics_l"])
        print("left distortion : ", oakd.calibr_info["dist_l"])

    frame_cnt = 0
    is_crosswalk = False
    while True:
        elev_flag = False
        start_time = time.time()
        oakd.wait_for_all_frames(decode_nn)
        # process previewout
        previewout = oakd.frameset['previewout']
        previewout_cpy = previewout.copy()
        camera = oakd.frameset['camera']
        depth_raw = oakd.frameset['depth_raw']
        window_name = 'previewout-' + camera

        if start_time - traffic_time_info["crosswalk"] > crosswalk_duration and is_crosswalk == True:
            is_crosswalk = False
            loop = asyncio.get_event_loop()
            task2 = loop.create_task(playaudio_name("audio_clips/disabling_crosswalk.mp3"))
            loop.run_until_complete(asyncio.gather(task2))

        openvino_app.perform_inference_road_seg(oakd.frameset['previewout'], inference_network, h, w, is_crosswalk)
        traffic_output = openvino_app.perform_inference_traffic_signs(oakd.frameset['previewout'], inference_network2,
                                                                      h_t, w_t)
        elev_img_probs = openvino_app.perform_inference_elev_img(previewout_cpy, elev_im_network, h, w)
        cropped_depth = depth_raw[500:700, 400:750]
        elev_depth_probs = openvino_app.perform_inference_elev_depth(cropped_depth, depth_network,
                                                                      h, w)
        pred_val_img = elev_img_probs[np.argmax(elev_img_probs)]
        pred_val_depth = elev_depth_probs[np.argmax(elev_img_probs)]
        if pred_val_img + pred_val_depth > 1.2:
            print("elevation = ",np.argmax(elev_img_probs))
            elev_flag = True
        # discard detections <= 0.5
        valid_detections = traffic_output[traffic_output[:, 2] > 0.5]
        traffic_labels = []
        traffic_angles = []
        if len(valid_detections):
            for detection in valid_detections:
                if openvino_app.TRAFFIC_DETECTOR_LABELS[int(detection[1])] == "traffic_sign":
                    ymin = int(oakd.frameset['previewout'].shape[0] * detection[4])
                    xmin = int(oakd.frameset['previewout'].shape[1] * detection[3])
                    ymax = int(oakd.frameset['previewout'].shape[0] * detection[6])
                    xmax = int(oakd.frameset['previewout'].shape[1] * detection[5])
                    height = ymax - ymin

                    tsign_image = oakd.frameset['previewout'][ymin:ymax, xmin:xmax]
                    cv2.rectangle(oakd.frameset['previewout'], (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                    tsign_class = openvino_app.perform_inference_signs_img(tsign_image, tsign_network2, 100, 300)
                    tsign_label = openvino_app.TRAFFIC_SIGN_CLASSES[tsign_class]
                    traffic_info[tsign_label] += 1
                    sign_time = time.time()
                    if (tsign_label == "stop_sign" or tsign_label == "stop_ahead") and traffic_info[tsign_label] > 25 and height > 26:
                        if sign_time - traffic_time_info["stop_sign"] > update_time_gap:
                            loop = asyncio.get_event_loop()
                            task2 = loop.create_task(playaudio_name("audio_clips/stop_sign.mp3"))
                            loop.run_until_complete(asyncio.gather(task2))
                            traffic_time_info["stop_sign"] = sign_time
                            is_crosswalk = True
                    if tsign_label == "pedestrian_sign" and traffic_info[tsign_label] > 25 and height > 26:
                        if sign_time - traffic_time_info[tsign_label] > update_time_gap:
                            loop = asyncio.get_event_loop()
                            task2 = loop.create_task(playaudio_name("audio_clips/pedestrian_crossing.mp3"))
                            loop.run_until_complete(asyncio.gather(task2))
                            traffic_time_info[tsign_label] = sign_time
                            is_crosswalk = True
                    if tsign_label == "sidewalk_closed" and traffic_info[tsign_label] > 25 and height > 26:
                        if sign_time - traffic_time_info[tsign_label] > update_time_gap:
                            loop = asyncio.get_event_loop()
                            task2 = loop.create_task(playaudio_name("audio_clips/sidewalk_closed.mp3"))
                            loop.run_until_complete(asyncio.gather(task2))
                            traffic_time_info[tsign_label] = sign_time
                    if is_crosswalk == True and sign_time - traffic_time_info["crosswalk"] > update_time_gap:
                        loop = asyncio.get_event_loop()
                        task2 = loop.create_task(playaudio_name("audio_clips/enabling_crosswalk.mp3"))
                        loop.run_until_complete(asyncio.gather(task2))
                        traffic_time_info["crosswalk"] = sign_time

                    cv2.putText(oakd.frameset['previewout'], tsign_label, (xmin, ymin),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0))
                    traffic_labels.append(tsign_label)
                else:
                    traffic_labels.append(openvino_app.TRAFFIC_DETECTOR_LABELS[int(detection[1])])
                mid_x = (detection[3] + detection[5]) / 2
                traffic_angle = clock_angle(mid_x)
                traffic_angles.append(traffic_angle)
        # print(traffic_labels)
        with open('traffic_labels.pkl', 'wb') as fh:
            pickle.dump(traffic_labels, fh)
        with open('traffic_angles.pkl', 'wb') as fh:
            pickle.dump(traffic_angles, fh)
        previewout_rgb = cv2.cvtColor(previewout, cv2.COLOR_BGR2RGB)
        oakd.nn_frame = show_nn(oakd.nnet_prev["entries_prev"][camera],
                                previewout, labels=labels, config=config)
        oakd_detections = oakd.nnet_prev["entries_prev"][camera]
        oakd_labels = []
        oakd_angles = []
        for e in oakd_detections:
            if labels[int(e[0]['label'])] not in unused_class:
                oakd_labels.append(labels[int(e[0]['label'])])
                mid_x = (e[0]['left'] + e[0]['right']) / 2
                oakd_angle = clock_angle(mid_x)
                oakd_angles.append(oakd_angle)
        with open('oakd_labels.pkl', 'wb') as fh:
            pickle.dump(oakd_labels, fh)
        with open('oakd_angles.pkl', 'wb') as fh:
            pickle.dump(oakd_angles, fh)
        oakd.add_fps(window_name, camera)
        cv2.rectangle(oakd.frameset["left"], (400, 500), (750, 700), (0, 0, 0))
        cv2.imshow("previewout", cv2.resize(oakd.frameset['previewout'], (640, 360)))
        previewout_video.write(cv2.resize(oakd.frameset['previewout'], (640, 360)))

        depth_f = copy.deepcopy(depth_raw)
        depth_copy = copy.deepcopy(depth_raw)
        depth_f = depth_f / 1000.0
        voice_counter += 1
        xy_coords[:, 2] = depth_f.flatten()
        xy_coords_2d = xy_coords.reshape(DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, 3)
        heights = [[200, 500], [0, 200]]
        widths = [[200, 950], [200, 950]]
        obstacle_grid = detect_left_right(xy_coords_2d, heights, widths,
                                          PROXIMITY_RANGE)
        # voice_counter = update_user(voice_counter, obstacle_grid)

        display_depth(depth_raw, oakd, window_name, heights, widths, elev_flag, obstacle_grid)
        points_within_range = xy_coords[xy_coords[:, 2] < 2]
        pcd = numpy_to_pcd(points_within_range)

        t_curr = time.time()
        if t_start + 1.0 < t_curr:
            t_start = t_curr
            oakd.update_params()

        key = cv2.waitKey(1)
        if key == ord('q'):
            traffic_labels = []
            oakd_labels = []
            with open('traffic_labels.pkl', 'wb') as fh:
                pickle.dump(traffic_labels, fh)
            with open('oakd_labels.pkl', 'wb') as fh:
                pickle.dump(oakd_labels, fh)
            break
        if key == ord('x'):
            collect_depth(depth_raw, frame_cnt, previewout_cpy)

        oakd.clear_frameset()
        frame_cnt += 1
        end_time = time.time()
        if end_time - traffic_info["last_t_stamp"] > 60:
            # flush traffic info
            flush_traffic_info(traffic_info, end_time)
            print("flushing traffic info", traffic_info)

        print("process_time = ", end_time-start_time)
    oakd.exit()


if __name__ == "__main__":
    main_2()
