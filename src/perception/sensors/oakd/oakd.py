import os
import sys
import time

import cv2
import numpy as np

import depthai
import consts.resource_paths

MAX_WAIT_TIME = 25


class OAK_D(object):
    BUFFER_SIZE = 30  # max buffer size for saving unsynced packets, usually size 2 is sufficient
    JPEG_REQ_RATE = 2  # request jpeg every other frame
    # buffer to store packets
    buffer = {}

    def __init__(self, config):
        self.frame_count = {}
        self.frame_count_prev = {}
        self.nnet_prev = {"entries_prev": {}, "nnet_source": {}}
        self.frame_count['nn'] = {}
        self.frame_count_prev['nn'] = {}
        self.previewout_rgb = None
        self._pipeline = None
        self.nn_cams = {'rgb', 'left', 'right'}
        self.nn_frame = None
        self.nnet_packets = None
        self.data_packets = None
        self.config = config
        self.right_bgr = None
        self.calibr_info = None
        self.jpeg_req_ctr = 0
        # latest single frameset
        self._frameset = {'previewout': None, 'right': None, 'depth_raw': None,
                          'packet_num': None, 'left': None, 'jpegout': None}
        self.frameset_count = 0

        for cam in self.nn_cams:
            self.nnet_prev["entries_prev"][cam] = []
            self.nnet_prev["nnet_source"][cam] = []
            self.frame_count['nn'][cam] = 0
            self.frame_count_prev['nn'][cam] = 0

        self.stream_names = [stream if isinstance(stream, str) else
                             stream['name'] for stream in config['streams']]
        self.stream_windows = []
        for stream in self.stream_names:
            if stream == 'previewout':
                for cam in self.nn_cams:
                    self.stream_windows.append(stream + '-' + cam)
            else:
                self.stream_windows.append(stream)

        for window in self.stream_windows:
            self.frame_count[window] = 0
            self.frame_count_prev[window] = 0

        # initialize device
        if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
            raise RuntimeError("Error initializing device. Try to reset it.")
        # create pipeline
        self._pipeline = depthai.create_pipeline(config=config)
        if self.pipeline is None:
            raise RuntimeError('Pipeline creation failed!')

        self.get_calibration("")  # "" looks in current directory

    @property
    def frameset(self):
        return self._frameset

    @property
    def pipeline(self):
        return self._pipeline

    def update_params(self):
        """
        taken from depthai's reference code
        """
        self.stream_windows = []
        for s in self.stream_names:
            if s == 'previewout':
                for cam in self.nn_cams:
                    self.stream_windows.append(s + '-' + cam)
                    self.frame_count_prev['nn'][cam] = self.frame_count['nn'][cam]
                    self.frame_count['nn'][cam] = 0
            else:
                self.stream_windows.append(s)
        for w in self.stream_windows:
            self.frame_count_prev[w] = self.frame_count[w]
            self.frame_count[w] = 0

    def fill_frameset(self):
        """
        fill frame information as we get them
        Note : jpegout does not have packet num, so picking the closest paccket
        TODO (OAK-D) : add packet num for jpegout
        """
        for packet in self.data_packets:
            if packet.stream_name not in self.stream_names:
                continue  # skip streams that were automatically added
            packet_data = packet.getData()
            if packet.stream_name == 'jpegout':  # does not work
                self.frameset['jpegout'] = cv2.imdecode(packet_data, cv2.IMREAD_COLOR)
                continue

            packet_num = packet.getMetadata().getSequenceNum()
            if packet_data is None:
                print('Invalid packet data!')
                continue

            if packet.stream_name != 'previewout' and packet_num not in self.buffer:
                self.buffer[packet_num] = {}
                self.buffer[packet_num]['right'] = None
                self.buffer[packet_num]['left'] = None
                self.buffer[packet_num]['depth_raw'] = None

            if packet.stream_name == 'previewout':
                self.frameset['previewout'] = self.bgr_from_packetdata(packet_data)
                self.frameset['camera'] = packet.getMetadata().getCameraName()
            if packet.stream_name == 'right':
                self.buffer[packet_num]['right'] = packet_data
            if packet.stream_name == 'left':
                self.buffer[packet_num]['left'] = packet_data
            if packet.stream_name.startswith('depth'):
                frame = packet_data
                if len(frame.shape) == 2:
                    if frame.dtype == np.uint16:  # uint16
                        self.buffer[packet_num]['depth_raw'] = frame

            if packet_num in self.buffer and self.is_pack_complete(self.buffer[packet_num]):
                self.frameset['depth_raw'] = self.buffer[packet_num]['depth_raw']
                self.frameset['right'] = self.buffer[packet_num]['right']
                self.frameset['left'] = self.buffer[packet_num]['left']
                self.frameset['packet_num'] = packet_num

                del self.buffer[packet_num]  # delete the packet once copied to frameset
        # health monitoring - flush buffer to remove stagnant packets
        if len(self.buffer.keys()) > self.BUFFER_SIZE:
            print("flushing buffer, buffer size = ", len(self.buffer.keys()))
            self.buffer = {}

    def wait_for_all_frames(self, decode_nn):
        """
        wait until all the frames are collected
        Note: requesting jpeg cuts fps rate largely
        """
        if 'jpegout' in self.config['streams']:
            if self.jpeg_req_ctr % self.JPEG_REQ_RATE == 0:
                depthai.request_jpeg()
                self.jpeg_req_ctr = 0
            self.jpeg_req_ctr += 1
        start_time = time.time()
        while not self.is_frame_complete():
            self.get_packets()
            self.decode_and_update(decode_nn)
            self.right_bgr = None
            self.fill_frameset()
            curr_time = time.time()
            if curr_time - start_time > MAX_WAIT_TIME:
                raise Exception('[Error] Time out waiting for frames')
        self.frameset_count += 1

    def is_frame_complete(self):
        """
        check if all the streams are collected
        Note: metaout is ignored for now
        """
        total_streams = 0
        for stream in self.config['streams']:
            if stream != 'metaout' and stream != 'jpegout':
                if self.frameset[stream] is not None:
                    total_streams += 1
        if 'jpegout' in self.config['streams']:
            return total_streams == len(self.config["streams"]) - 2  # ignore metaout and jpegout
        else:
            return total_streams == len(self.config["streams"]) - 1  # ignore metaout

    def is_pack_complete(self, packet):
        """
        making sure all the frames are present
        Note: 'previewout' stream is not synced as it starts earlier than other streams
        """
        total_streams = 0
        for stream in self.config['streams']:
            if stream != 'metaout' and stream != 'previewout':
                if stream in packet and packet[stream] is not None:
                    total_streams += 1
        if 'jpegout' in self.config['streams']:
            return total_streams == len(self.config['streams']) - 3  # ignore metaout and previewout and jpegout
        else:
            return total_streams == len(self.config['streams']) - 2  # ignore metaout and previewout

    def get_packets(self):
        """
        get nn and data packets from pipeline
        """
        self.nnet_packets, self.data_packets = self.pipeline.get_available_nnet_and_data_packets()

    def decode_and_update(self, decode_nn):
        """
        decode nn packets and update others information
        """
        for _, nnet_packet in enumerate(self.nnet_packets):
            camera = nnet_packet.getMetadata().getCameraName()
            self.nnet_prev["nnet_source"][camera] = nnet_packet
            self.nnet_prev["entries_prev"][camera] = decode_nn(nnet_packet, config=self.config)
            self.frame_count['metaout'] += 1
            self.frame_count['nn'][camera] += 1

    def bgr_from_packetdata(self, packet_data):
        """
        convert data packet to opencv format
        """
        data0 = packet_data[0, :, :]
        data1 = packet_data[1, :, :]
        data2 = packet_data[2, :, :]
        frame = cv2.merge([data0, data1, data2])
        return frame

    def add_fps(self, window_name, camera):
        """
        cv2.putText(self.nn_frame, "fps: " + str(self.frame_count_prev[window_name]), (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0))
        cv2.putText(self.nn_frame, "NN fps: " + str(self.frame_count_prev['nn'][camera]),
                    (2, self.nn_frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
        """
        pass

    def get_calibration(self, calibr_path=""):
        try:
            self.calibr_info = np.load(os.path.join(calibr_path, 'calibr_info.npz'))
        except FileNotFoundError:
            print("calibr_info not found, continuing without calibration information")
            return
        print("calibration information loaded")

    def clear_frameset(self):
        """
        reset all the frames
        """
        self._frameset = {'previewout': None, 'right': None, 'depth_raw': None,
                          'packet_num': None, 'left': None, 'jpegout': None}

    def exit(self):
        """
        delete the the pipeline and de-init device
        """
        del self._pipeline
        depthai.deinit_device()