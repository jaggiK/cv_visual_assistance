from time import time

import numpy as np
import cv2
import open3d as o3d

from oakd_config import config, decode_nn, show_nn, labels
from oakd import OAK_D


def display_depth(frame, oakd, window_name):
    frame = (65535 // frame).astype(np.uint8)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    cv2.putText(frame, "depth", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
    cv2.putText(frame, "fps: " + str(oakd.frame_count_prev[window_name]), (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 255)
    cv2.imshow("depth_raw", cv2.resize(frame, (640, 360)))


def numpy_to_pcd(points, image=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if image is not None:
        pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
    return pcd


def main():
    t_start = time()
    # get OAK-D instance - does init, creates pipeline + other initialization
    oakd = OAK_D(config)
    if oakd.calibr_info is not None:
        print("right intrinsics : ", oakd.calibr_info["intrinsics_r"])
        print("right distortion : ", oakd.calibr_info["dist_r"])
        print("left intrinsics : ", oakd.calibr_info["intrinsics_l"])
        print("left distortion : ", oakd.calibr_info["dist_l"])

    while True:
        oakd.wait_for_all_frames(decode_nn)
        # process previewout
        previewout = oakd.frameset['previewout']
        camera = oakd.frameset['camera']
        window_name = 'previewout-' + camera
        oakd.nn_frame = show_nn(oakd.nnet_prev["entries_prev"][camera],
                                previewout, labels=labels, config=config)
        oakd.add_fps(window_name, camera)
        cv2.imshow(window_name, oakd.nn_frame)

        # process right
        if oakd.frameset['right'] is not None:
            cv2.imshow("right", cv2.resize(oakd.frameset['right'], (640, 360)))

        # process left
        if oakd.frameset['left'] is not None:
            cv2.imshow("left", cv2.resize(oakd.frameset['left'], (640, 360)))

        # process left
        if oakd.frameset['jpegout'] is not None:
            cv2.imshow("jpegout", cv2.resize(oakd.frameset['jpegout'], (640, 360)))

        # process depth
        frame = oakd.frameset['depth_raw']
        display_depth(frame, oakd, window_name)
        oakd.frame_count[window_name] += 1

        t_curr = time()
        if t_start + 1.0 < t_curr:
            t_start = t_curr
            # print("metaout fps: " + str(oakd.frame_count_prev["metaout"]))
            oakd.update_params()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        oakd.clear_frameset()
    oakd.exit()


if __name__ == "__main__":
    main()