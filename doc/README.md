# Welcome to visual assitance system using OAK-D and ege AI devices


Hardware: 

1. OAK-D camera kit
2. Neural compute sticks 
3. USB enable GPS; Tested with VK-162 G-Mouse USB enabled GPS
4. Laptop with Ubuntu 18.04
5. Power supply for camera kit. Tested with Anker 10000mAH power bank 

Software requirements:

Python Gen1 DepthAI for OAK-D sensor. Installation steps can be found here - https://github.com/luxonis/depthai. Note that OAK-D sensor is required.

OpenVINO for model optimization and pretrained models. Installation link: https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html

Open3D for point cloud processing, verison 0.10.0.0

Python GPSD package for GPS interface

Vosk speech recognition package. Installation steps here : https://github.com/alphacep/vosk-api

Festival Text-to-speech. https://github.com/usc-isi-i2/festival-text-to-speech-service/blob/master/README.md

Machine learning packages : Tensorflow (1.14.0), Tensofrlow Lite, Pytorch (Optional)

While this may be a large repo. The dataset and training scripts are separated for modularity and simplicity

## Datasets

Object detection model dataset 300x300 can be found here : https://drive.google.com/drive/folders/1HgLOO-HA3YntmjhF0FZvdrJhK-FmV-2C?usp=sharing

Object detection model dataset 400x400 can be found here : https://drive.google.com/drive/folders/15UoVreqdEoQ1WpBIwM1z9L6P2A9vQ2om?usp=sharing

Object detection model dataset 400x400 can be found here : https://drive.google.com/drive/folders/1_M5pVEadWE7oauOXqa5TBtrvrJzqdpny?usp=sharing

Combined traffic dataset can be found here : https://drive.google.com/drive/folders/1K5g6XJF6RIMA_J82POtwkdnpHfuRRBa1?usp=sharing

## Training scripts

Please follow this notebook for training object detection model : https://colab.research.google.com/drive/1cdTI74VYkdYkNlaIDGg9uCCxfIaFV2RI?usp=sharing
The notebook also includes steps for optimizing models using OpenVINO.

`cv_visual_assistance/src/perception/sensors/oakd/train_depth_image.py` script can be used to train MinniVGGNet for training on depth images for elevational changes
















