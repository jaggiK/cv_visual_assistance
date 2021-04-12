from playsound import playsound
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--filename', help="audio clip filename")
args = parser.parse_args()
filename = args.filename
filename = filename.replace("'", "")
print("------ ", filename)
time.sleep(2)
playsound(args.filename)