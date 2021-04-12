"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

VPU_DEVICE_NAME = 'VPU'
MYRIAD_DEVICE_NAME = 'MYRIAD'
HDDL_DEVICE_NAME = 'HDDL'
FPGA_DEVICE_NAME = 'FPGA'
CPU_DEVICE_NAME = 'CPU'
GPU_DEVICE_NAME = 'GPU'
HETERO_DEVICE_NAME = 'HETERO'
MULTI_DEVICE_NAME = 'MULTI'
UNKNOWN_DEVICE_TYPE = 'UNKNOWN'

XML_EXTENSION = '.xml'
BIN_EXTENSION = '.bin'

XML_EXTENSION_PATTERN = '*' + XML_EXTENSION

IMAGE_EXTENSIONS = ['JPEG', 'JPG', 'PNG', 'BMP']
BINARY_EXTENSIONS = ['BIN']

DEVICE_DURATION_IN_SECS = {
    CPU_DEVICE_NAME: 60,
    GPU_DEVICE_NAME: 60,
    VPU_DEVICE_NAME: 60,
    MYRIAD_DEVICE_NAME: 60,
    HDDL_DEVICE_NAME: 60,
    FPGA_DEVICE_NAME: 120,
    UNKNOWN_DEVICE_TYPE: 120
}

DEVICE_NIREQ_ASYNC = {
    CPU_DEVICE_NAME: 2,
    GPU_DEVICE_NAME: 2,
    VPU_DEVICE_NAME: 4,
    MYRIAD_DEVICE_NAME: 4,
    HDDL_DEVICE_NAME: 100,
    FPGA_DEVICE_NAME: 3,
    UNKNOWN_DEVICE_TYPE: 1
}
