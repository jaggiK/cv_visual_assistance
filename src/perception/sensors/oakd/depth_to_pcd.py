
import numpy as np
import open3d as o3d
import glob
import cv2

# intrinic params, taken from "python3 test.py" output
intrinsics_right_cam = np.load('/home/jaggi/cruisecrafter_synced/cruisecrafter/src/perception/sensors/oakd/depthai/intrinsics.npz')['M2']
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

PROXIMITY_RANGE = 1.5
pcd = o3d.geometry.PointCloud()

def numpy_to_pcd(points, image=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if image is not None:
        pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
    return pcd


import os
DATA_DIR = "/home/jaggi/depth_image_dataset"
def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        print(class_map)
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            # train_points.append(trimesh.load(f).sample(num_points))
            if f.endswith(".jpg"):
                continue
            print("train = ", f)
            depth_f = np.load(f)["full"]
            depth_f = depth_f / 1000.0
            xy_coords[:, 2] = depth_f.flatten()
            points_within_range = xy_coords[xy_coords[:, 2] < 2]
            pcd = numpy_to_pcd(points_within_range)
            # o3d.visualization.draw_geometries([pcd])

            xy_coords_2d = xy_coords.reshape(DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, 3)
            cropped_box = xy_coords_2d[500:700, 400:750]
            cropped_region = cropped_box.reshape(-1, 3)
            cropped_region_temp = cropped_region.copy()
            cropped_region_temp[cropped_region_temp[:, 2] > 3] = 0.0
            train_points.append(cropped_region_temp)

            train_labels.append(i)

        for f in test_files:
            # test_points.append(trimesh.load(f).sample(num_points))
            if f.endswith(".jpg"):
                continue
            print("test = ", f)
            depth_f = np.load(f)["full"]
            depth_f = depth_f / 1000.0
            xy_coords[:, 2] = depth_f.flatten()
            points_within_range = xy_coords[xy_coords[:, 2] < 2]
            pcd = numpy_to_pcd(points_within_range)
            # o3d.visualization.draw_geometries([pcd])

            xy_coords_2d = xy_coords.reshape(DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, 3)
            cropped_box = xy_coords_2d[500:700, 400:750]
            cropped_region = cropped_box.reshape(-1, 3)
            cropped_region_temp = cropped_region.copy()
            cropped_region_temp[cropped_region_temp[:, 2] > 3] = 0.0
            test_points.append(cropped_region_temp)


            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )




import time
for f in glob.glob("/home/jaggi/depth_image_dataset/up/train/*.npz"):
    print(f)
    depth_f = np.load(f)["full"]
    # depth_f = depth_f/1000.0
    frame_copy = (65535 // depth_f).astype(np.uint8)
    frame_copy = cv2.applyColorMap(frame_copy, cv2.COLORMAP_HOT)
    xy_coords[:, 2] = depth_f.flatten()
    points_within_range = xy_coords[xy_coords[:, 2] < 2]
    pcd = numpy_to_pcd(points_within_range)
    # o3d.visualization.draw_geometries([pcd])

    xy_coords_2d = xy_coords.reshape(DEPTH_FRAME_HEIGHT, DEPTH_FRAME_WIDTH, 3)
    cropped_box = xy_coords_2d[500:700, 400:750]
    cropped_region = cropped_box.reshape(-1, 3)
    cropped_region_temp = cropped_region.copy()

    points_within_range = cropped_region_temp[cropped_region_temp[:, 2] < 2]
    cropped_pcd_temp = numpy_to_pcd(points_within_range)
    # print("normal pcd pts = ", len(np.asarray(cropped_pcd_temp.points)))
    # o3d.visualization.draw_geometries([cropped_pcd_temp])
    # cropped_pcd_temp.estimate_normals()

    # cropped_region_temp[cropped_region_temp[:, 2] > 2.5] = 0.0
    # cropped_pcd_temp = numpy_to_pcd(cropped_region_temp)
    # print("zeroed pcd pts = ", len(np.asarray(cropped_pcd_temp.points)))
    # # o3d.visualization.draw_geometries([cropped_pcd_temp])
    # print(len(cropped_region_temp[cropped_region_temp[:, 2] == 0.0]))
    print(cropped_pcd_temp)
    cropped_region = cropped_region[cropped_region[:, 2] < 2]
    cropped_pcd = numpy_to_pcd(cropped_region)
    # o3d.visualization.draw_geometries([cropped_pcd])
    
    image = cv2.imread(f.replace(".npz", ".jpg"))
    image = cv2.resize(image, (640, 360))
    crop_img = image[250:350, 200:375]
    print(image.shape)
    cv2.imshow("img", image)
    cv2.imshow("depth", frame_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# import os
# dir = "/home/jaggi/depth_image_dataset_up2"
# new_dir = "/home/jaggi/depth_image_up/"
#
# for f in glob.glob(dir+"/*.npz"):
#     new_f = f.split("/")[-1].replace(".npz", "_up2.npz")
#     path = new_dir + new_f
#     os.rename(f, path)
#
# for f in glob.glob(dir+"/*.jpg"):
#     new_f = f.split("/")[-1].replace(".jpg", "_up2.jpg")
#     path = new_dir + new_f
#     os.rename(f, path)
# import os
# import numpy as np
# dir = "/home/jaggi/depth_image_dataset/down"
# for jpg_file in glob.glob(dir+"/*.jpg"):
#     np_file = jpg_file.replace(".jpg", ".npz")
#     prob = np.random.randint(100)
#     if prob < 20:
#         new_jpg_file = jpg_file.replace(dir, dir+"/test")
#         new_np_file = np_file.replace(dir, dir + "/test")
#     else:
#         new_jpg_file = jpg_file.replace(dir, dir+"/train")
#         new_np_file = np_file.replace(dir, dir + "/train")
#     os.rename(jpg_file, new_jpg_file)
#     os.rename(np_file, new_np_file)

# dataset = parse_dataset(num_points=1000)