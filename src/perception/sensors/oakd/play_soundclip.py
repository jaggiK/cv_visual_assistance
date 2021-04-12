from playsound import playsound
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--filename', help="audio clip filename")

args = parser.parse_args()
# print(args.filename)
obstacle_clips = ["left.mp3", "right.mp3", "center.mp3", "front.mp3"]
obstacles_info = args.filename
print("playsound = ", obstacles_info)
print(type(args.filename))
obstacles_info = obstacles_info.replace(" ", "")
obstacles_info = obstacles_info.replace(",", "")
obstacles_info = obstacles_info.replace("[", "")
obstacles_info = obstacles_info.replace("]", "")
obstacles_info = obstacles_info.replace("'", "")
top = False
f = open("playsound_text.txt", "a")
f.write(obstacles_info)
f.close()


def convert_to_2D(obstacles_info):
    grid_2d = []
    for i, ele in enumerate(obstacles_info):
        if i % 4 == 0:
            continue
        grid_2d.append(int(ele))
    grid_2d = np.array(grid_2d)
    grid_2d = grid_2d.reshape(-1, 3)
    return grid_2d



def analyze_grid(grid_2d):
    bottom = np.sum(grid_2d[0,:])
    top = np.sum(grid_2d[1, :])
    left = np.sum(grid_2d[:,0])
    right = np.sum(grid_2d[:,1])
    center = np.sum(grid_2d[:,2])

    if bottom == 0 and top > 0:
        playsound("audio_clips/top.mp3")
    if right and left and center:
        playsound("audio_clips/front.mp3")
    elif center and right:
        playsound("audio_clips/center.mp3")
        playsound("audio_clips/right.mp3")
    elif center and left:
        playsound("audio_clips/center.mp3")
        playsound("audio_clips/left.mp3")
    elif left and right:
        playsound("audio_clips/left.mp3")
        playsound("audio_clips/right.mp3")
    elif right:
        playsound("audio_clips/right.mp3")
    elif left:
        playsound("audio_clips/left.mp3")
    elif center:
        playsound("audio_clips/center.mp3")


grid_2d = convert_to_2D(obstacles_info)
print(grid_2d)
analyze_grid(grid_2d)

