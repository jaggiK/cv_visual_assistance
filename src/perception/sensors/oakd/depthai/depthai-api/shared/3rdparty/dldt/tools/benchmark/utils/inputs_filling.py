"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import cv2
import numpy as np

from glob import glob

from .constants import IMAGE_EXTENSIONS, BINARY_EXTENSIONS
from .logging import logger


def is_image(blob):
    if blob.layout != "NCHW":
        return False
    channels = blob.shape[1]
    return channels == 3


def is_image_info(blob):
    if blob.layout != "NC":
        return False
    channels = blob.shape[1]
    return channels >= 2


def get_inputs(path_to_input, batch_size, input_info, requests):
    input_image_sizes = {}
    for key in input_info.keys():
        if is_image(input_info[key]):
            input_image_sizes[key] = (input_info[key].shape[2], input_info[key].shape[3])
        logger.info("Network input '{}' precision {}, dimensions ({}): {}".format(key,
                                                                                  input_info[key].precision,
                                                                                  input_info[key].layout,
                                                                                  " ".join(str(x) for x in
                                                                                           input_info[key].shape)))

    images_count = len(input_image_sizes.keys())
    binaries_count = len(input_info) - images_count

    image_files = list()
    binary_files = list()

    if path_to_input:
        image_files = get_files_by_extensions(path_to_input, IMAGE_EXTENSIONS)
        image_files.sort()
        binary_files = get_files_by_extensions(path_to_input, BINARY_EXTENSIONS)
        binary_files.sort()

    if (len(image_files) == 0) and (len(binary_files) == 0):
        logger.warn("No input files were given: all inputs will be filled with random values!")
    else:
        binary_to_be_used = binaries_count * batch_size * len(requests)
        if binary_to_be_used > 0 and len(binary_files) == 0:
            logger.warn("No supported binary inputs found! Please check your file extensions: {}".format(
                ",".join(BINARY_EXTENSIONS)))
        elif binary_to_be_used > len(binary_files):
            logger.warn(
                "Some binary input files will be duplicated: {} files are required, but only {} were provided".format(
                    binary_to_be_used, len(binary_files)))
        elif binary_to_be_used < len(binary_files):
            logger.warn(
                "Some binary input files will be ignored: only {} files are required from {}".format(binary_to_be_used,
                                                                                                     len(binary_files)))

        images_to_be_used = images_count * batch_size * len(requests)
        if images_to_be_used > 0 and len(image_files) == 0:
            logger.warn("No supported image inputs found! Please check your file extensions: {}".format(
                ",".join(IMAGE_EXTENSIONS)))
        elif images_to_be_used > len(image_files):
            logger.warn(
                "Some image input files will be duplicated: {} files are required, but only {} were provided".format(
                    images_to_be_used, len(image_files)))
        elif images_to_be_used < len(image_files):
            logger.warn(
                "Some image input files will be ignored: only {} files are required from {}".format(images_to_be_used,
                                                                                                    len(image_files)))

    requests_input_data = []
    for request_id in range(0, len(requests)):
        logger.info("Infer Request {} filling".format(request_id))
        input_data = {}
        keys = list(input_info.keys())
        for key in keys:
            if is_image(input_info[key]):
                # input is image
                if len(image_files) > 0:
                    input_data[key] = fill_blob_with_image(image_files, request_id, batch_size, keys.index(key),
                                                           len(keys), input_info[key].shape)
                    continue

            # input is binary
            if len(binary_files):
                input_data[key] = fill_blob_with_binary(binary_files, request_id, batch_size, keys.index(key),
                                                        len(keys), input_info[key].shape)
                continue

            # most likely input is image info
            if is_image_info(input_info[key]) and len(input_image_sizes) == 1:
                image_size = input_image_sizes[list(input_image_sizes.keys()).pop()]
                logger.info("Fill input '" + key + "' with image size " + str(image_size[0]) + "x" +
                            str(image_size[1]))
                input_data[key] = fill_blob_with_image_info(image_size, input_info[key].shape)
                continue

            # fill with random data
            logger.info("Fill input '{}' with random values ({} is expected)".format(key, "image" if is_image(
                input_info[key]) else "some binary data"))
            input_data[key] = fill_blob_with_random(input_info[key].precision, input_info[key].shape)

        requests_input_data.append(input_data)

    return requests_input_data


def get_files_by_extensions(path_to_input, extensions):
    get_extension = lambda file_path: file_path.split(".")[-1].upper()

    input_files = list()
    if os.path.isfile(path_to_input):
        files = [os.path.normpath(path_to_input)]
    else:
        path = os.path.join(path_to_input, '*')
        files = glob(path, recursive=True)
    for file in files:
        file_extension = get_extension(file)
        if file_extension in extensions:
                input_files.append(file)

    return input_files


def fill_blob_with_image(image_paths, request_id, batch_size, input_id, input_size, shape):
    images = np.ndarray(shape)
    image_index = request_id * batch_size * input_size + input_id
    for b in range(batch_size):
        image_index %= len(image_paths)
        image_filename = image_paths[image_index]
        logger.info('Prepare image {}'.format(image_filename))
        image = cv2.imread(image_filename)

        new_im_size = tuple(shape[2:])
        if image.shape[:-1] != new_im_size:
            logger.warn("Image is resized from ({}) to ({})".format(image.shape[:-1], new_im_size))
            image = cv2.resize(image, new_im_size)

        if image.shape[0] != shape[2]:
            image = image.transpose((2, 1, 0))
        else:
            image = image.transpose((2, 0, 1))
        images[b] = image

        image_index += input_size
    return images


def fill_blob_with_binary(binary_paths, request_id, batch_size, input_id, input_size, shape):
    binaries = np.ndarray(shape)
    binary_index = request_id * batch_size * input_size + input_id
    for b in range(batch_size):
        binary_index %= len(binary_paths)
        binary_filename = binary_paths[binary_index]

        binary_file_size = os.path.getsize(binary_filename)
        input_size = np.prod(shape) / batch_size
        if input_size != binary_file_size:
            raise Exception(
                "File {} contains {} bytes but network expects {}".format(binary_filename, binary_file_size, input_size))

        with open(binary_filename, 'r') as f:
            binary_data = f.read()

        binaries[b] = binary_data
        binary_index += input_size

    return binaries


def fill_blob_with_image_info(image_size, shape):
    im_info = np.ndarray(shape)
    for b in range(shape[0]):
        for i in range(shape[1]):
            im_info[b][i] = image_size[i] if i in [0, 1] else 1

    return im_info


def fill_blob_with_random(precision, shape):
    if precision == "FP32":
        return np.random.rand(*shape).astype(np.float32)
    elif precision == "FP16":
        return np.random.rand(*shape).astype(np.float16)
    elif precision == "I32":
        return np.random.rand(*shape).astype(np.int32)
    elif precision == "U8":
        return np.random.rand(*shape).astype(np.uint8)
    elif precision == "I8":
        return np.random.rand(*shape).astype(np.int8)
    elif precision == "U16":
        return np.random.rand(*shape).astype(np.uint16)
    elif precision == "I16":
        return np.random.rand(*shape).astype(np.int16)
    else:
        raise Exception("Input precision is not supported: " + precision)
