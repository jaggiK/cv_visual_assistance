import json

import consts.resource_paths

is_mobile_net = True
calc_dist_to_bb = True
if is_mobile_net:
    cnn_model_path = consts.resource_paths.nn_resource_path + "mobilenet-ssd" + "/" + "mobilenet-ssd"
    blob_file = cnn_model_path + ".blob"
    suffix = ""
    if calc_dist_to_bb:
        suffix = "_depth"
    blob_file_config = cnn_model_path + suffix + ".json"
    blob_file_config = cnn_model_path + suffix + ".json"
    from depthai_helpers.mobilenet_ssd_handler import decode_mobilenet_ssd, show_mobilenet_ssd

    decode_nn = decode_mobilenet_ssd
    show_nn = show_mobilenet_ssd

    with open(blob_file_config) as f:
        data = json.load(f)

    try:
        labels = data['mappings']['labels']
    except:
        labels = None
        print("Labels not found in json!")


config={
    'streams': ['previewout', 'metaout', 'depth_raw'],
    'ai': {
        "blob_file": blob_file,
        "blob_file_config": blob_file_config,
        'calc_dist_to_bb': calc_dist_to_bb,
        'shaves' : 7,
        'cmx_slices' : 7,
        'NN_engines' : 1,
        'keep_aspect_ratio': True
    },
    'app':
        {
            'sync_video_meta_streams': True,
        },
    'depth':
        {
            'calibration_file': consts.resource_paths.calib_fpath,
            'padding_factor': 0.3,
            'depth_limit_m': 10.0,  # In meters, for filtering purpose during x,y,z calc
            'confidence_threshold': 0.81,
            # Depth is calculated for bounding boxes with confidence higher than this number
        },
    'camera':
        {
            'rgb':
                {
                    # 3840x2160, 1920x1080
                    # only UHD/1080p/30 fps supported for now
                    'resolution_h': 1080,#args['rgb_resolution'],
                    'fps': 5,
                },
            'mono':
                {
                    # 1280x720, 1280x800, 640x400 (binning enabled)
                    'resolution_h': 720,#args['mono_resolution'],
                    'fps': 5,
                },
        },

    'app':
        {
            'sync_video_meta_streams': False,
        },
}