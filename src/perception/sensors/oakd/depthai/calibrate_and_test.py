#!/usr/bin/env python3

import os
import signal
import subprocess
import atexit
import sys
import time

global p
global return_code

p=None

def cleanup():
    if(p is not None):
        print('Stopping subprocess with pid: ', str(p.pid))
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print('Stopped!')

args=""

for arg in sys.argv[1:]:
    args+="'"+arg+"' "

calibrate_cmd = "python3 calibrate.py " + args
test_cmd = """python3 depthai.py -co '{"streams": [{"name": "depth_raw", "max_fps": 12.0}]}'"""


atexit.register(cleanup)

p = subprocess.Popen(calibrate_cmd, shell=True, preexec_fn=os.setsid) 
p.wait()
return_code = p.returncode
p=None
print("Return code:"+str(return_code))
time.sleep(3)

if(return_code == 0):
    p = subprocess.Popen(test_cmd, shell=True, preexec_fn=os.setsid) 
    p.wait()
    return_code = p.returncode
    p=None
    print("Return code:"+str(return_code))


exit(return_code)