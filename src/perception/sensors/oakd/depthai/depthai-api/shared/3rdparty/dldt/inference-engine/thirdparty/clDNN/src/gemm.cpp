/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>
#include <utility>
#include <algorithm>

namespace cldnn {
primitive_type_id gemm::type_id() {
    static primitive_type_base<gemm> instance;
    return &instance;
}

layout gemm_inst::calc_output_layout(gemm_node const& node) {
    auto prim = node.get_primitive();

    auto input0_layout = node.input(0).get_output_layout();
    auto input1_layout = node.input(1).get_output_layout();
    bool transpose_input0 = prim->transpose_input0;
    bool transpose_input1 = prim->transpose_input1;

    auto M = !transpose_input0 ? input0_layout.size.spatial[1] : input0_layout.size.spatial[0];
    auto N = !transpose_input1 ? input1_layout.size.spatial[0] : input1_layout.size.spatial[1];

    auto output_size = input0_layout.size;

    for (size_t i = 1; i < node.inputs_count(); ++i) {
        auto input_layout = node.input(i).get_output_layout();
        output_size = tensor::max(output_size, input_layout.size);
    }

    output_size.spatial[0] = N;
    output_size.spatial[1] = M;
    auto output_type = input0_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && prim->output_data_type)
        output_type = *prim->output_data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    auto output_format = input0_layout.format;

    return layout(output_type, output_format, output_size, prim->output_padding);
}

std::string gemm_inst::to_string(gemm_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto alpha = desc->alpha;
    auto beta = desc->beta;
    auto transpose_input0 = desc->transpose_input0 ? " true" : "false";
    auto transpose_input1 = desc->transpose_input1 ? " true" : "false";
    std::stringstream primitive_description;

    json_composite gemm_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        gemm_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    gemm_info.add("alpha", alpha);
    gemm_info.add("beta", beta);
    gemm_info.add("trasnpose_input0", transpose_input0);
    gemm_info.add("transpose_input1", transpose_input1);
    node_info->add("gemm info", gemm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gemm_inst::typed_primitive_inst(network_impl& network, gemm_node const& node) : parent(network, node) {
    auto input0_layout = node.input(0).get_output_layout();
    auto input1_layout = node.input(1).get_output_layout();
    bool transpose_input0 = node.get_primitive()->transpose_input0;
    bool transpose_input1 = node.get_primitive()->transpose_input1;

    auto transposed_x0 = input0_layout.size.spatial[0];
    auto transposed_y0 = input0_layout.size.spatial[1];

    if (transpose_input0) {
        std::swap(transposed_x0, transposed_y0);
    }

    auto transposed_x1 = input1_layout.size.spatial[0];
    auto transposed_y1 = input1_layout.size.spatial[1];

    if (transpose_input1) {
        std::swap(transposed_x1, transposed_y1);
    }

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input 0 internal dimension size",
                          transposed_x0,
                          "Input 1 internal dimension size",
                          transposed_y1,
                          "");

    if (node.inputs_count() == 3) {
        auto input2_layout = node.input(2).get_output_layout();

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Input 0 external dimension size",
                              transposed_y0,
                              "Input 2 rows number",
                              input2_layout.size.spatial[1],
                              "");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Input 1 external dimension size",
                              transposed_x1,
                              "Input 2 columns number",
                              input2_layout.size.spatial[0],
                              "");
    }
}
}  // namespace cldnn
