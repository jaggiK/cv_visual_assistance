#### Thanks to Udacity for the OpenVINO course, most of this code/logic is from their tutorials 

Example to download models (make sure OpenVINO is installed)

```
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

sudo ./downloader.py --name human-pose-estimation-0001 -o /home/jaggi/openvino_models
(download with model precision )

sudo ./downloader.py --name text-detection-0004 --precisions FP16 -o /home/jaggi/openvino_models

semantic segmentation - road, crosswalk, person, bike, motorcycle, vegetation, car etc (20 classes)
./downloader.py --name semantic-segmentation-adas-0001 -o /home/jaggi/openvino_models

pedestrain attributes - gender, has bag, has backpack, has long hair etc. (7 classes)
./downloader.py --name person-attributes-recognition-crossroad-0230 -o /home/jaggi/openvino_models

car detection model
./downloader.py --name vehicle-detection-adas-0002 -o /home/jaggi/openvino_models

car + bike detection
./downloader.py --name person-vehicle-bike-detection-crossroad-0078 -o /home/jaggi/openvino_models
```

## Deploy Your First Edge App - Solution
### `app.py`

Within `app.py`, the main work is just to call `preprocess_input` and `handle_output` in
the correct locations. You can feed `args.m` into these so they receive the model type,
and will return the appropriate preprocessing or output handling function. You can then feed
the input image or output in as applicable.


The rest of the app will then create the relevant output images so you can see the Inference
Engine at work with the Pre-Trained Models you've worked with throughout the lesson.

Here are the commands I used to run the app for each:

Note : option `-c` is ignored, will be removed, `outputs` should be created at same level as `images`

Classification example: (retaining `-c` here, just for reference )
```
python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/jaggi/openvino_models/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
```

Text Detection example:
```
python app.py -i "images/sign.jpg" -t "TEXT" -m "/home/jaggi/openvino_models/intel/text-detection-0004/FP16/text-detection-0004.xml" -c "dummy"
```

Pose Estimation example:
```
python app.py -i "images/sitting-on-car.jpg" -t "POSE" -m "/home/jaggi/openvino_models/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml" -c "dummy"
```

Pedestrian detection example
```
python app.py -i images/ped.jpg -t "PER" -m "/home/jaggi/openvino_models/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml" -c "dummy"
```

Semantic Segmentation example
```
python app.py -i images/bikeonsidewalk2.png -t "SEM" -m "/home/jaggi/openvino_models/intel/semantic-segmentation-adas-0001/FP16/semantic-segmentation-adas-0001.xml" -c "dummy"
```

Car + bike detection example
```
python app.py -i images/car.jpg -t "VEH_BIKE" -m "/home/jaggi/openvino_models/intel/person-vehicle-bike-detection-crossroad-0078/FP16/person-vehicle-bike-detection-crossroad-0078.xml" -c "dummy"
```

Car detection
```
python app.py -i images/car4.jpg -t "VEH" -m "/home/jaggi/openvino_models/intel/vehicle-detection-adas-0002/FP16/vehicle-detection-adas-0002.xml" -c "dummy"
```

Passing folder as argument

```
python app.py -t "SEG" -m "/home/jaggi/openvino_models/intel/semantic-segmentation-adas-0001/FP16/semantic-segmentation-adas-0001.xml" -c "dummy" -f /home/jaggi/vis_dataset/
```