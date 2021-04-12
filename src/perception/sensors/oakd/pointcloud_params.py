import copy

import numpy as np
import cv2
import asyncio
import open3d as o3d

# intrinic params, taken from "python3 test.py" output
intrinsics_right_cam = np.load('depthai/intrinsics.npz')['M2']
print(intrinsics_right_cam)
k_mat = np.array([[1.016701, -0.008633, -15.218214],
                  [0.013645, 1.013096, -22.569859],
                  [0.000008, 0.000003, 1.000000], ])
k_mat = intrinsics_right_cam/1000.0
k_inv = np.linalg.inv(k_mat)
DEPTH_FRAME_HEIGHT = 720
DEPTH_FRAME_WIDTH = 1280
# generate image indices
image_indices = [[i, j, 1] for i in range(0, DEPTH_FRAME_HEIGHT) for j in range(0, DEPTH_FRAME_WIDTH)]
image_indices = np.array(image_indices)

xy_coords = np.matmul(k_inv, image_indices.transpose())
xy_coords = xy_coords.T.reshape(DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, 3)
xy_coords = xy_coords.reshape(DEPTH_FRAME_HEIGHT * DEPTH_FRAME_WIDTH, 3)
xy_coords = xy_coords / 1000.0

RELIABLE_DEPTH = 0.8
DEPTH_THRESH = 1.2


def detect_left_right(xy_coords_2d, heights, widths, proximity):

    obstacle_grid = {}
    for index, (height, width) in enumerate(zip(heights, widths)):
        l_x_length = 0.0
        l_y_length = 0.0
        r_y_length = 0.0
        r_x_length = 0.0
        c_x_length = 0.0
        c_y_length = 0.0
        obstacle_right = False
        obstacle_center = False
        obstacle_left = False
        box = xy_coords_2d[height[0]: height[1], width[0]:width[1]]
        right_box = box[:, int(box.shape[1] * 3 / 4):]
        left_box = box[:, : int(box.shape[1] * 1 / 4)]
        center_box = box[:, int(box.shape[1] * 1 / 4) : int(box.shape[1] * 3 / 4)]

        center_region = center_box.reshape(-1, 3)
        center_region = center_region[center_region[:, 2] < 2]
        num_points_center = np.count_nonzero(center_region[:, 2] < proximity)

        right_region = right_box.reshape(-1, 3)
        right_region = right_region[right_region[:, 2] < 2]
        num_points_right = np.count_nonzero(right_region[:, 2] < proximity)

        left_region = left_box.reshape(-1, 3)
        left_region = left_region[left_region[:, 2] < 2]
        num_points_left = np.count_nonzero(left_region[:, 2] < proximity)

        if num_points_center > 1200:
            obstacle_center = True
            center_pcd = o3d.geometry.PointCloud()
            center_pcd.points = o3d.utility.Vector3dVector(center_region)
            center_pcd = center_pcd.voxel_down_sample(voxel_size=0.12)
            center_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            # o3d.visualization.draw_geometries([right_pcd])
            points = o3d.utility.Vector3dVector(center_region)
            aabbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(center_pcd.points)
            # print(aabbox.get_axis_aligned_bounding_box())
            bbox_points = aabbox.get_box_points()
            min_pt = bbox_points[0]
            max_pt = bbox_points[4]
            c_x_length = np.abs(max_pt[0] - min_pt[0])
            c_y_length = np.abs(max_pt[2] - min_pt[2])
            lines = [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 7],
                [1, 6],
                [2, 7],
                [2, 5],
                [3, 5],
                [3, 6],
                [4, 6],
                [5, 4],
                [4, 7],
            ]
            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            center_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # o3d.visualization.draw_geometries([center_pcd, line_set])
        if num_points_right > 1200:
            obstacle_right = True
            right_pcd = o3d.geometry.PointCloud()
            right_pcd.points = o3d.utility.Vector3dVector(right_region)
            right_pcd = right_pcd.voxel_down_sample(voxel_size=0.02)
            right_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            # o3d.visualization.draw_geometries([right_pcd])
            points = o3d.utility.Vector3dVector(right_region)
            aabbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(right_pcd.points)
            # print(aabbox.get_axis_aligned_bounding_box())
            bbox_points = aabbox.get_box_points()
            min_pt = bbox_points[0]
            max_pt = bbox_points[4]
            r_x_length = np.abs(max_pt[0] - min_pt[0])
            r_y_length = np.abs(max_pt[2] - min_pt[2])
            lines = [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 7],
                [1, 6],
                [2, 7],
                [2, 5],
                [3, 5],
                [3, 6],
                [4, 6],
                [5, 4],
                [4, 7],
            ]
            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            right_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # o3d.visualization.draw_geometries([right_pcd , line_set])
            # print(np.asarray(bbox_points))
        if num_points_left > 1200:
            obstacle_left = True
            left_pcd = o3d.geometry.PointCloud()
            left_pcd.points = o3d.utility.Vector3dVector(left_region)
            left_pcd = left_pcd.voxel_down_sample(voxel_size=0.02)
            left_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            # o3d.visualization.draw_geometries([right_pcd])
            points = o3d.utility.Vector3dVector(left_region)
            aabbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(left_pcd.points)
            # print(aabbox.get_axis_aligned_bounding_box())
            bbox_points = aabbox.get_box_points()
            min_pt = bbox_points[0]
            max_pt = bbox_points[4]
            l_x_length = np.abs(max_pt[0] - min_pt[0])
            l_y_length = np.abs(max_pt[1] - min_pt[1])
            lines = [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 7],
                [1, 6],
                [2, 7],
                [2, 5],
                [3, 5],
                [3, 6],
                [4, 6],
                [5, 4],
                [4, 7],
            ]
            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            left_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # o3d.visualization.draw_geometries([left_pcd, line_set])
        obstacle_grid[index] = [obstacle_left, obstacle_right, obstacle_center]
        obstacle_grid[str(index)] = {"left":[l_x_length, l_y_length], "right":[r_x_length, r_y_length], "center":[c_x_length, c_y_length]}
    return obstacle_grid


def process_pointcloud(frame, packet, oak_handle, voice_counter):
    if len(frame.shape) == 2:
        print(packet.stream_name)
        if frame.dtype == np.uint8:  # grayscale
            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            cv2.putText(frame, "fps: " + str(oak_handle.frame_count_prev[window_name]), (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255))
        else:  # uint16
            obs_right = False
            obs_left = False
            obstacle = False
            CLOSER_PROXIMITY = 1.5
            voice_counter += 1
            depth_f = copy.deepcopy(frame)
            depth_f = depth_f / 1000.0

            xy_coords[:, 2] = depth_f.flatten()
            xy_coords_2d = xy_coords.reshape(DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, 3)
            heights = [200, 500]
            widths = [300, 800]
            centre_box = xy_coords_2d[heights[0]: heights[1], widths[0]:widths[1]]
            right_box = centre_box[:, int(centre_box.shape[1] * 3 / 4):]
            left_box = centre_box[:, : int(centre_box.shape[1] * 1 / 4)]

            centre_region = centre_box.reshape((heights[1] - heights[0]) * (widths[1] - widths[0]), 3)
            temp = centre_region[centre_region[:, 2] < 2]
            num_points_middle = np.count_nonzero(temp[:, 2] < CLOSER_PROXIMITY)

            right_region = right_box.reshape(-1, 3)
            temp = right_region[right_region[:, 2] < 2]
            num_points_right = np.count_nonzero(temp[:, 2] < CLOSER_PROXIMITY)

            left_region = left_box.reshape(-1, 3)
            temp = left_region[left_region[:, 2] < 2]
            num_points_left = np.count_nonzero(temp[:, 2] < CLOSER_PROXIMITY)

            if num_points_middle > 2000:
                obstacle = True

            if num_points_right > 1500:
                obs_right = True

            if num_points_left > 1500:
                obs_left = True

            if voice_counter > 30 and (obstacle or obs_right):
                loop = asyncio.get_event_loop()
                # result = loop.run_until_complete(speak_word("obstacle"))
                update_str = "obstacle"
                if obs_left:
                    update_str = update_str + " left"
                if obs_right:
                    update_str = update_str + " right"
                task1 = loop.create_task(speak_word(update_str))
                loop.run_until_complete(asyncio.gather(task1))
                voice_counter = 0
                # tts_p.speak("obstaacle..")
            print(num_points_middle)
            # print(xy_coords[0])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(temp)
            frame = (65535 // frame).astype(np.uint8)
            closer_points = np.count_nonzero(frame < RELIABLE_DEPTH * 1000)
            # colorize depth map, comment out code below to obtain grayscale
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            cv2.rectangle(frame, (widths[0], heights[0]), (widths[1], heights[1]), (0, 255, 0))
            cv2.imshow("depth_cust", frame)
            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
            cv2.putText(frame, "fps: " + str(oak_handle.frame_count_prev[window_name]), (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, 255)
