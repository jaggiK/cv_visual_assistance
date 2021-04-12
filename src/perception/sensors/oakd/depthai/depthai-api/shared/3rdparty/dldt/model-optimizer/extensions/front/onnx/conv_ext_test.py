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

import unittest

import numpy as np
import onnx

from extensions.front.onnx.conv_ext import ConvTransposeFrontExtractor
from mo.utils.unittest.graph import build_graph
from mo.graph.graph import Node
from mo.utils.error import Error


class ConvTransposeONNXExtractorTest(unittest.TestCase):
    @staticmethod
    def _create_node(attrs: dict):
        pb = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], **attrs)
        graph = build_graph({'node_0': {'pb': pb}}, [])
        return Node(graph, 'node_0')

    @staticmethod
    def _base_attrs():
        # Commonly used attributes in the tests
        # Each test takes these ones and then adds/modifies/deletes particular fields
        return (
            # test input ONNX attributes
            dict(
                pads=[1, 2, 3, 4],
                kernel_shape=[5, 6]
            ),
            # reference output Node attributes
            dict(
                type='Deconvolution',
                pad=[[0, 0], [0, 0], [1, 3], [2, 4]],
                bias_term=None,
                output_shape=None,
                output_padding=None,
                dilation=None,
                stride=None,
                output_spatial_shape=None,
                group=1
            )
        )

    @staticmethod
    def _extract(inp):
        node = __class__._create_node(inp)
        ConvTransposeFrontExtractor.extract(node)
        return node.graph.node[node.id]

    def _match(self, out, ref):
        for key in ref.keys():
            status = out[key] == ref[key]
            if type(status) in [list, np.ndarray]:
                status = np.all(status)
            self.assertTrue(status, 'Mismatch for field {}, observed: {}, expected: {}'.format(key, out[key], ref[key]))

    def test_all_valid_default(self):
        inp, ref = self._base_attrs()
        del inp['pads']
        del ref['pad']
        out = self._extract(inp)
        self._match(out, ref)

    def test_most_used(self):
        inp, ref = self._base_attrs()
        out = self._extract(inp)
        self._match(out, ref)

    def test_dilation(self):
        inp, ref = self._base_attrs()
        inp['dilations'] = [10, 11]
        ref['dilation'] = [1, 1, 10, 11]
        out = self._extract(inp)
        self._match(out, ref)

    def test_stride(self):
        inp, ref = self._base_attrs()
        inp['strides'] = [12, 13]
        ref['stride'] = [1, 1, 12, 13]
        out = self._extract(inp)
        self._match(out, ref)

    def test_group(self):
        inp, ref = self._base_attrs()
        inp['group'] = 14
        ref['group'] = 14
        out = self._extract(inp)
        self._match(out, ref)

    def test_auto_pad_supported(self):
        inp, ref = self._base_attrs()
        del inp['pads']
        inp['auto_pad'] = 'SAME_UPPER'

        ref['auto_pad'] = 'same_upper'
        del ref['pad']

        out = self._extract(inp)
        self._match(out, ref)

    def test_pads_not_even_invalid(self):
        inp, ref = self._base_attrs()
        inp['pads'] = [1, 2, 3]
        with self.assertRaisesRegex(Error, '.*pads.*not correct.*'):
            out = self._extract(inp)

    def test_missing_kernel_shape_not_supported(self):
        inp, ref = self._base_attrs()
        del inp['kernel_shape']
        with self.assertRaisesRegex(Error, '.*kernel_shape.*not supported.*'):
            out = self._extract(inp)

    def test_output_padding(self):
        inp, ref = self._base_attrs()
        inp['output_padding'] = [19, 20]
        ref['output_padding'] = [0, 0, 19, 20]
        out = self._extract(inp)
        self._match(out, ref)
