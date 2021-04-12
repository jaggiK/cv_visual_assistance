Download SSD model from here:
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

`tar -xvzf <path_to_dowloaded_model>`

`cd <unzipped file`>`


#convert tf model to Openvino's IR format
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`

This produces IR format file - `frozen_inference_graph.xml` (network architecture) 
and `frozen_inference_graph.bin` (weights)

Example usage:
`python main.py -m /home/jaggi/openvino_tutorial/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -ct 0.6 -c BLUE`

This will produce `output-custom.mp4` file with car being detected on the input file `test_video.mp4`