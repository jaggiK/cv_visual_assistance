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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging as log

from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs
from mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.loader import load_onnx_model, protobuf2nx
from mo.pipeline.common import get_ir_version
from mo.utils import class_registration
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def driver(argv: argparse.Namespace):
    model_proto = load_onnx_model(argv.input_model)
    model_graph = model_proto.graph  # pylint: disable=no-member
    # print(model_graph)
    # assert len(model_graph) == 1, "An ONNX model contains more than 1 graph: unsupported"
    log.debug("Number of nodes in graph_def: {}".format(len(model_graph.node)))
    log.debug("Number of all input ports (not true inputs) in graph_def: {}".format(len(model_graph.input)))
    log.debug("Number of initializers in graph_def: {}".format(len(model_graph.initializer)))
    log.debug("Number of real inputs in graph_def: {}".format(len(model_graph.input) - len(model_graph.initializer)))
    update_extractors_with_extensions(onnx_op_extractors)

    try:
        graph = protobuf2nx(model_proto)
        log.debug("Number of nodes in NX graph: {}".format(graph.number_of_nodes()))
        graph.__setattr__('name',
                          argv.model_name if argv.model_name else model_proto.graph.name)  # pylint: disable=no-member
        graph.graph['layout'] = 'NCHW'
        graph.graph['cmd_params'] = argv
        graph.graph['fw'] = 'onnx'
        graph.graph['feature_dim'] = 1 if graph.graph['layout'] == 'NCHW' else 3
        graph.graph['ir_version'] = get_ir_version(argv)

    except Exception as e:
        raise Error(
            'Cannot pre-process ONNX graph after reading from model file "{}". ' \
            'File is corrupt or has unsupported format. Details: {}. ' +
            refer_to_faq_msg(44),
            argv.input_model,
            str(e)
        ) from e
    graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
    extract_node_attrs(graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))

    # --------------------------------- LOAD END ------------------------------------------------------

    class_registration.apply_replacements(graph, [
        class_registration.ClassType.FRONT_REPLACER,
        class_registration.ClassType.MIDDLE_REPLACER,
        class_registration.ClassType.BACK_REPLACER
    ])

    return graph
