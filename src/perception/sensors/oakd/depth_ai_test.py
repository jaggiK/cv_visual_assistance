import copy
from time import time
import subprocess

import numpy as np
import cv2
import open3d as o3d
import asyncio

from oakd_config import config, blob_file_config, decode_nn, show_nn, labels
from oakd import OAK_D
from pointcloud_params import xy_coords, RELIABLE_DEPTH, DEPTH_THRESH, DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, \
    detect_left_right, intrinsics_right_cam
from playsound import playsound
import os

PROXIMITY_RANGE = 1.5


async def speak_word(word):
    subprocess.Popen('echo ' + word + '|festival --tts', shell=True)


async def play_sound(audio_name):
    # print("cur_dir = ", os.getcwd())
    subprocess.Popen(["python3", "play_soundclip.py", "-f", "{}".format(audio_name)])#play_soundclip.py")#--filename {}".format(audio_name))
    # playsound(audio_name)

def update_user(voice_counter, obstacle_l, obstacle_c, obstacle_r):
    if voice_counter > 30 and (obstacle_l or obstacle_c or obstacle_r):
        loop = asyncio.get_event_loop()
        update_str = "obstacle"
        if obstacle_l:
            update_str = update_str + " left"
        if obstacle_r:
            update_str = update_str + " right"
        if obstacle_c:
            update_str = update_str + "center"

        # task1 = loop.create_task(speak_word(update_str))
        # loop.run_until_complete(asyncio.gather(task1))
        print(int(obstacle_l), int(obstacle_r), int(obstacle_c))
        task1 = loop.create_task(play_sound([int(obstacle_l), int(obstacle_r), int(obstacle_c)]))
        loop.run_until_complete(asyncio.gather(task1))
        """
        if obstacle_l:
            task1 = loop.create_task(play_sound("left.mp3"))
            loop.run_until_complete(asyncio.gather(task1))
        if obstacle_r:
            task2 = loop.create_task(play_sound("right.mp3"))
            loop.run_until_complete(asyncio.gather(task2))
        if obstacle_c:
            task3 = loop.create_task(play_sound("center.mp3"))
            loop.run_until_complete(asyncio.gather(task3))
        """
        voice_counter = 0
    return voice_counter


def display_depth(frame, oakd, window_name, heights, widths):
    frame = (65535 // frame).astype(np.uint8)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    cv2.rectangle(frame, (widths[0], heights[0]), (widths[1], heights[1]), (0, 255, 0))
    cv2.imshow("depth_cust", frame)
    cv2.putText(frame, "depth", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
    cv2.putText(frame, "fps: " + str(oakd.frame_count_prev[window_name]), (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 255)


def numpy_to_pcd(points, image=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if image is not None:
        pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
    return pcd


def main():
    t_start = time()
    voice_counter = 0
    # get OAK-D instance - does init, creates pipeline + other initialization
    oakd = OAK_D(config)
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
        frame_bgr = oakd.frameset['right']
        # oakd.right_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
        # cv2.imshow("right", cv2.resize(oakd.frameset['right'], (640, 360)))

        # process depth
        frame = oakd.frameset['depth_raw']
        depth_f = copy.deepcopy(frame)
        depth_f = depth_f / 1000.0
        voice_counter += 1
        xy_coords[:, 2] = depth_f.flatten()
        xy_coords_2d = xy_coords.reshape(DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, 3)
        heights = [200, 500]
        widths = [300, 800]
        obstacle_l, obstacle_c, obstacle_r = detect_left_right(xy_coords_2d, heights, widths,
                                                               PROXIMITY_RANGE)
        voice_counter = update_user(voice_counter, obstacle_l, obstacle_c, obstacle_r)

        display_depth(frame, oakd, window_name, heights, widths)
        points_within_range = xy_coords[xy_coords[:, 2] < 2]
        pcd = numpy_to_pcd(points_within_range)
        # o3d.visualization.draw_geometries([pcd])
        # o3d.visualization.draw_geometries([downpcd])
        # downpcd = pcd.voxel_down_sample(voxel_size=0.01)
        oakd.frame_count[window_name] += 1

        t_curr = time()
        if t_start + 1.0 < t_curr:
            t_start = t_curr
            # print("metaout fps: " + str(oakd.frame_count_prev["metaout"]))
            oakd.update_params()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            name = str(oakd.frameset_count)
            o3d.io.write_point_cloud(f"/home/jaggi/pcds/{name}.ply", pcd)
            cv2.imwrite(f"/home/jaggi/pcds/{name}_right_image.jpg", oakd.frameset['right'])
            np.savez(f"/home/jaggi/pcds/{name}_depth", Depth=depth_f, intrinsics=intrinsics_right_cam)
        oakd.clear_frameset()
    oakd.exit()


if __name__ == "__main__":
    main()

# pcd = o3d.io.read_point_cloud("/home/jaggi/pcds/1.ply")
# o3d.visualization.draw_geometries([pcd])