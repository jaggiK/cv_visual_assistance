#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer
import os
from playsound import playsound
import subprocess
import signal

if not os.path.exists("speech_model"):
    print(
        "Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

import pyaudio
import ast
from semantic_seg import main_2
import sys

sys.path.append("../../../")
import pickle

async def speak_word(word):
    subprocess.Popen('echo ' + word + '|festival --tts', shell=True)



async def start_perception():
    bash_command = "gnome-terminal -- python3 semantic_seg.py &"
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()

async def start_perception_2():
    main_2()


model = Model("speech_model")
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()
voice_interface_active = False
save_location = False
describe = False
started = False
from shutil import copyfile

speech_process = None
perception_process = None
gps_process = None

while True:
    data = stream.read(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        rec_speech = {}
        rec_speech = rec.Result()
        rec_speech = ast.literal_eval(rec_speech)
        if rec_speech["text"] == "okay buddy" or rec_speech["text"] == "okay birdie" or rec_speech["text"] == "okay betty":
            playsound("audio_clips/entry.mp3")
            voice_interface_active = True
        voice_interface_active = True
        if voice_interface_active and not started and (rec_speech["text"] == "start" or rec_speech["text"] == "stark"):
            speech_process = subprocess.Popen(["gnome-terminal", "--disable-factory", "-x", "python3", "speech_actions.py"], preexec_fn=os.setpgrp)
            perception_process = subprocess.Popen(["gnome-terminal", "--disable-factory", "-x", "python3", "semantic_seg.py"], preexec_fn=os.setpgrp)
            gps_process = subprocess.Popen(["gnome-terminal", "--disable-factory", "-x", "python3", "gps_parser.py"], preexec_fn=os.setpgrp)
            started = True
        if voice_interface_active and started and rec_speech["text"] == "exit":
            print("trying to quit..")
            os.killpg(int(speech_process.pid), signal.SIGKILL)
            os.killpg(int(perception_process.pid), signal.SIGKILL)
            os.killpg(int(gps_process.pid), signal.SIGKILL)
            started = False
        with open('rec_speech.pkl', 'wb') as fh:
            if rec_speech["text"] == "":
                rec_speech["text"] = "NIL"
            pickle.dump(rec_speech, fh)

        copyfile("rec_speech.pkl", "rec_speech_2.pkl")
        print(rec_speech, type(rec_speech))


