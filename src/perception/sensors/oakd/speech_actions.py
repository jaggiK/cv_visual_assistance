import pickle
from playsound import playsound
import asyncio
import subprocess
from semantic_seg import main_2
import os
from shutil import copy
import time
import glob
import geopy.distance
from twilio.rest import Client

async def speak_word(word):
    subprocess.Popen('echo ' + word + '|festival --tts', shell=True)

def speak_word_sync(word):
    subprocess.Popen('echo ' + word + '|festival --tts', shell=True)

async def start_perception():
    bash_command = "python3 semantic_seg.py"
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()

async def start_perception_2():
    main_2()


voice_interface_active = False
save_location = False
describe = False
started = False

target = "rec_speech_2.pkl"
last_set_time = time.time()
TIME_THRESHOLD = 10 #seconds
GPS_DIR = "/home/jaggi/cruisecrafter_synced/cruisecrafter/src/perception/sensors/oakd"
while True:
    rec_speech = {}
    rec_speech["text"] = "NIL"
    if os.path.getsize(target) > 0:
        try:
            with open(target, "rb") as fh:
                rec_speech = pickle.load(fh)
        except EOFError:
            print("continuing")
            continue

    rec_word = rec_speech["text"]
    if rec_word == "okay buddy" or rec_word == "okay birdie" or rec_word == "okay betty":
        playsound("audio_clips/entry.mp3")
        voice_interface_active = True
    if rec_word == "stop describe":
        describe = False
    voice_interface_active = True
    try:
        with open("obstacle_info.pkl", "rb") as ft:
            obstacle_info = pickle.load(ft)
        if obstacle_info:
            pass#continue
        if voice_interface_active:
            if save_location:
                # speak_word(rec_speech["text"])
                curr_time = time.time()
                print(curr_time, last_set_time)
                if curr_time - last_set_time > TIME_THRESHOLD:
                    loop = asyncio.get_event_loop()
                    task_read_saved = loop.create_task(speak_word(rec_word))
                    loop.run_until_complete(asyncio.gather(task_read_saved))
                    try:
                        fname = rec_word.replace(" ","_")
                        fname = "gps_" + fname + ".pkl"
                        copy("gps_coords.pkl", fname)
                        gps_coords = pickle.load(ft)
                        print("saved as ", rec_word)
                    except:
                        gps_coords = []
                    save_location = False
            if rec_word == "start":
                print("starting")
                # start_perception_2()
                # loop = asyncio.get_event_loop()
                # task1 = loop.create_task(start_perception())
                # print("task = ", task1)
                # if not started:
                #     os.system("gnome-terminal -x python3 semantic_seg.py &")
                #     started = True
                #     describe = False
                #     save_location = False
                # loop.run_until_complete(asyncio.gather(task1))
            if rec_word == "save location" or rec_word == "same location":
                curr_time = time.time()
                print(curr_time, last_set_time)
                if curr_time - last_set_time > TIME_THRESHOLD:
                    save_location = True
                    playsound("audio_clips/name_please.mp3")
                    last_set_time = curr_time
            if rec_word == "locate" or rec_word == "located":
                curr_time = time.time()
                if curr_time - last_set_time > TIME_THRESHOLD:
                    playsound("audio_clips/locating.mp3")
                    try:
                        with open("curr_gps_coords.pkl", "rb") as ft:
                            curr_gps_coords = pickle.load(ft)
                            print("curr gps = ", curr_gps_coords)
                    except:
                        curr_gps_coords = []
                    #get saved locations
                    for gf in glob.glob(GPS_DIR + "/*.pkl"):
                        if gf.split("/")[-1].startswith("gps"):
                            print(gf)
                            with open(gf, "rb") as fg:
                                gps_coords = pickle.load(fg)
                            dist = geopy.distance.geodesic(gps_coords, curr_gps_coords).meters
                            location_name = gf.split("/")[-1].replace("gps", "").replace(".pkl", "").replace("_", " ")
                            # loop = asyncio.get_event_loop()
                            # task_read_saved = loop.create_task(speak_word(location_name))
                            # loop.run_until_complete(asyncio.gather(task_read_saved))
                            time.sleep(1)
                            speak_word_sync(location_name)
                            time.sleep(1)
                            speak_word_sync(str(round(dist,2)))
                            time.sleep(3)
                            speak_word_sync("meters")
                            time.sleep(1)
                            print(gps_coords)



                    last_set_time = curr_time
            if rec_word == "describe" or rec_word == "described" or describe == True:
                describe = True
                try:
                    with open("traffic_labels.pkl", "rb") as ft:
                        traffic_labels = pickle.load(ft)
                except:
                    traffic_labels = []

                try:
                    with open("oakd_labels.pkl", "rb") as fo:
                        oakd_labels = pickle.load(fo)
                except:
                        oakd_labels = []
                try:
                    with open("traffic_angles.pkl", "rb") as fo:
                        traffic_angles = pickle.load(fo)
                except:
                        traffic_angles = []
                try:
                    with open("oakd_angles.pkl", "rb") as fo:
                        oakd_angles = pickle.load(fo)
                except:
                        oakd_angles = []
                if rec_word == "stop":
                    describe = False

                # playsound("describing.mp3")
                for label, angle in zip(traffic_labels, traffic_angles):
                    print(label)
                    playsound("audio_clips/" + label + ".mp3")
                    playsound("audio_clips/" + angle + ".mp3")
                    playsound("audio_clips/" + "oclock.mp3")
                for label, angle in zip(oakd_labels, oakd_angles):
                    print(label)
                    playsound("audio_clips/" + label + ".mp3")
                    playsound("audio_clips/" + angle + ".mp3")
                    playsound("audio_clips/" + "oclock.mp3")

            if rec_word == "exit":
                # playsound("exit.mp3")
                # voice_interface_active = False
                pass
            if rec_word == "share location" or rec_word == "shared location":
                curr_time = time.time()
                if curr_time - last_set_time > TIME_THRESHOLD:
                    try:
                        with open("curr_gps_coords.pkl", "rb") as ft:
                            curr_gps_coords = pickle.load(ft)
                            print("curr gps = ", curr_gps_coords)
                    except:
                        curr_gps_coords = None
                    if curr_gps_coords is not None:
                        location_str = "location = "+ str(curr_gps_coords[0]) + ", " + str(curr_gps_coords[1])
                        account_sid = 'AC9298d5d2c6bec5608bf477b58cf57d83'
                        auth_token = '45017941c09868091e2800eaa4f000b4'
                        client = Client(account_sid, auth_token)
                        message = client.messages \
                            .create(
                            body=location_str,
                            from_='+17148315421',
                            to='+17062962343'
                        )
                        last_set_time = curr_time
    except Exception as e:
        continue