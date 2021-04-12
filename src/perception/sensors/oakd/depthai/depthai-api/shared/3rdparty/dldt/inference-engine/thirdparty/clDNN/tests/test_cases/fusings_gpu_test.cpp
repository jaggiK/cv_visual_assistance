/*
// Copyright (c) 2019 Intel Corporation
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
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include "api/input_layout.hpp"
#include "api/convolution.hpp"
#include "api/quantize.hpp"
#include "api/topology.hpp"
#include "api/tensor.hpp"
#include "api/network.hpp"
#include "api/eltwise.hpp"
#include "api/fully_connected.hpp"
#include "api/gemm.hpp"
#include "api/binary_convolution.hpp"
#include "api/engine.hpp"
#include "api/data.hpp"

#include "test_utils/test_utils.h"

#include <cmath>

using namespace cldnn;
using namespace tests;

struct bc_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    tensor stride;
    tensor pad;
    tensor dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

struct gemm_test_params {
    std::vector<tensor> in_shapes;
    tensor kernel;
    tensor pad;
    data_types data_type_in0;
    data_types data_type_in1;
    data_types data_type_in2;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

template<typename T>
class BaseFusingTest : public ::testing::TestWithParam<T> {
public:
    cldnn::engine engine;
    cldnn::topology topology;
    cldnn::build_options bo_fused;
    cldnn::build_options bo_not_fused;

    float tolerance = 0.0f;

    static const int min_random = -200;
    static const int max_random = 200;

    void SetUp() override {
        bo_fused.set_option(build_option::optimize_data(true));
        bo_not_fused.set_option(build_option::optimize_data(false));
    }

    void compare(network& not_fused, network& fused, T& p) {
        auto outputs_ref = not_fused.execute();
        auto outputs_fused = fused.execute();

        auto get_reorders_count = [](network& net) -> size_t {
            size_t count = 0;
            for (auto& pi : net.get_primitives_info()) {
                if (pi.type_id == "reorder") {
                    count++;
                }
            }
            return count;
        };

        size_t reorders_count_fused = get_reorders_count(fused);
        size_t reorders_count_not_fused = get_reorders_count(not_fused);

        // Subtract reorders count to handle execution in different layouts when input/output reorders can be added in the graph
        ASSERT_EQ(fused.get_executed_primitives().size() - reorders_count_fused, p.expected_fused_primitives);
        ASSERT_EQ(not_fused.get_executed_primitives().size() - reorders_count_not_fused, p.expected_not_fused_primitives);
        ASSERT_EQ(outputs_ref.size(), outputs_fused.size());
        ASSERT_EQ(outputs_ref.size(), size_t(1));

        auto output_not_fused_prim = outputs_ref.begin()->second.get_memory();
        auto output_fused_prim = outputs_fused.begin()->second.get_memory();
        if (output_not_fused_prim.get_layout().data_type == data_types::f32) {
            auto ref = output_not_fused_prim.pointer<float>();
            auto output_ptr = output_fused_prim.pointer<float>();
            for (size_t i = 0; i < output_fused_prim.get_layout().count(); i++) {
                ASSERT_NEAR(ref[i], output_ptr[i], tolerance) << "i = " << i;
            }
        } else {
            auto ref = output_not_fused_prim.pointer<int16_t>();
            auto output_ptr = output_fused_prim.pointer<int16_t>();
            for (size_t i = 0; i < output_fused_prim.get_layout().count(); i++) {
                ASSERT_NEAR(float16_to_float32(ref[i]), float16_to_float32(output_ptr[i]), tolerance) << "i = " << i;
            }
        }
    }

    cldnn::memory get_mem(cldnn::layout l) {
        auto prim = memory::allocate(engine, l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8 || l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec = generate_random_1d<uint8_t>(s.count(), min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec = generate_random_1d<uint16_t>(s.count(), min_random, max_random);
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), min_random, max_random);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory get_mem(cldnn::layout l, float fill_value) {
        auto prim = memory::allocate(engine, l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec(s.count() / 32, static_cast<int32_t>(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec(s.count(), float32_to_float16(fill_value));
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec(s.count(), fill_value);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory get_mem(cldnn::layout l, int min, int max) {
        auto prim = memory::allocate(engine, l);
        tensor s = l.size;
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }
        else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }
        else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    layout get_output_layout(T& p) {
        return layout{ p.data_type, p.input_format, p.out_shape };
    }

    layout get_weights_layout(T& p, const int32_t split = 1) {
        return layout{p.weights_type, p.weights_format, tensor{p.out_shape.feature[0] / split,
                                                               static_cast<int32_t>(p.in_shape.feature[0] / p.groups),
                                                               p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]}};
    }

    layout get_bias_layout(T& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }

    layout get_weights_zp_layout(T& p) {
        return layout{ p.weights_type, p.default_format, tensor{p.out_shape.feature[0], 1, 1, 1} };
    }

    layout get_activations_zp_layout(T& p) {
        return layout{ p.data_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1} };
    }

    layout get_single_element_layout(T& p) {
        return layout{ p.default_type, p.default_format, tensor{1, 1, 1, 1} };
    }
};

class WeightsPrimitiveFusingTest : public ::BaseFusingTest<bc_test_params> {
public:

    void execute(bc_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology, bo_not_fused);
        network network_fused(this->engine, this->topology, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(bc_test_params& p) {
        auto pad = p.pad.negate();
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{pad_} };
    }

    layout get_per_channel_layout(bc_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};


class GemmFusingTest : public ::BaseFusingTest<gemm_test_params> {
public:

    void execute(gemm_test_params& p) {
        auto input0_prim = get_mem(get_input_layout(p, 0));
        auto input1_prim = get_mem(get_input_layout(p, 1));

        network network_not_fused(this->engine, this->topology, bo_not_fused);
        network network_fused(this->engine, this->topology, bo_fused);
        network_fused.set_input_data("input0", input0_prim);
        network_not_fused.set_input_data("input0", input0_prim);
        network_fused.set_input_data("input1", input1_prim);
        network_not_fused.set_input_data("input1", input1_prim);
        if (p.in_shapes.size() > 2) {
            auto input2_prim = get_mem(get_input_layout(p, 2));
            network_fused.set_input_data("input2", input2_prim);
            network_not_fused.set_input_data("input2", input2_prim);
        }

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gemm_test_params& p, int in_no) {
        auto pad = p.pad.negate();
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        if (in_no == 0)
            return layout{ p.data_type_in0, p.input_format, p.in_shapes.at(0), padding{pad_} };
        else if (in_no == 1)
            return layout{ p.data_type_in1, p.input_format, p.in_shapes.at(1), padding{pad_} };
        else
            return layout{ p.data_type_in2, p.input_format, p.in_shapes.at(2), padding{pad_} };
    }

    layout get_per_channel_layout(gemm_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.in_shapes.at(0).feature[0], 1, 1} };
    }
};

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_FP32_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_2 {1, 16, 4, 5}, {1, 32, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx_f16, data_types::f32, format::o_i_yx_i16_o16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx_f16, data_types::f32, format::o_i_yx_i16_o16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f32, format::bfyx_f16, data_types::f32,  format::oiyx_o16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_5 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_6 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfzyx_f16, data_types::f32, format::o_i_zyx_i16_o16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_7 {1, 16, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfzyx_f16, data_types::f32, format::o_i_zyx_i16_o16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_8 {1, 32, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::bfzyx_f16, data_types::f32, format::o_i_zyx_i16_o16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_9 {1, 32, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::bfzyx_f16, data_types::f32, format::o_i_zyx_i16_o16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_10 {32, 16, 4, 5, 4}, {32, 32, 4, 5, 4}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfzyx_b16f16, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_CONV_FP16_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_2 {1, 16, 4, 5}, {1, 32, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx_f16, data_types::f16, format::o_i_yx_i16_o16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx_f16, data_types::f16, format::o_i_yx_i16_o16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f16, format::bfyx_f16, data_types::f16,  format::oiyx_o16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_5 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx, data_types::i8, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_6 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfzyx_f16, data_types::f16, format::o_i_zyx_i16_o16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_7 {1, 16, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfzyx_f16, data_types::f16, format::o_i_zyx_i16_o16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_8 {1, 32, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f16, format::bfzyx_f16, data_types::f16, format::o_i_zyx_i16_o16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_9 {1, 32, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f16, format::bfzyx_f16, data_types::f16, format::o_i_zyx_i16_o16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_10 {32, 16, 4, 5, 4}, {32, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfzyx_b16f16, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

#define CASE_CONV_U8S8_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_2 {1, 15, 5, 5}, {1, 30, 3, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_4 {1, 17, 4, 5}, {1, 17, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 17, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_5 {1, 16, 5, 5}, {1, 32, 5, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_CONV_S8S8_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_2 {1, 15, 5, 5}, {1, 30, 3, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_4 {1, 17, 4, 5}, {1, 17, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 17, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_5 {1, 16, 5, 5}, {1, 32, 5, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_CONV3D_U8S8_1 {1, 15, 5, 4, 5}, {1, 30, 3, 2, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_2 {1, 15, 5, 5, 5}, {1, 30, 3, 3, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_3 {1, 16, 5, 4, 5}, {1, 32, 5, 4, 5}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_4 {1, 17, 5, 4, 5}, {1, 17, 5, 4, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, -1}, tensor{1}, 17, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_CONV3D_S8S8_1 {1, 15, 5, 4, 5}, {1, 30, 3, 2, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_2 {1, 15, 5, 5, 5}, {1, 30, 3, 3, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_3 {1, 16, 5, 4, 5}, {1, 32, 5, 4, 5}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_4 {1, 17, 5, 4, 5}, {1, 17, 5, 4, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{{0, 0, -1, -1, -1}, 0}, tensor{1}, 17, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_BIN_CONV1 {1, 16, 4, 5}, {1, 16, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV2 {1, 16, 4, 5}, {1, 30, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV3 {1, 184, 12, 21}, {1, 224, 12, 21}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx


#define CASE_FC_FP32_1 {1, 1, 3, 1}, {1, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_2 {2, 1, 3, 1}, {2, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::yxfb, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3 {2, 32, 1, 1}, {2, 16, 1, 1}, {16, 32, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_FC_U8S8_1 {1, 1, 3, 1}, {1, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_2 {2, 1, 3, 1}, {2, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3 {2, 32, 1, 1}, {2, 16, 1, 1}, {16, 32, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_3IN_S8S8_1 {{1, 1, 2, 2}, {1, 1, 2, 2}, {1, 1, 2, 2}}, tensor{1}, tensor{0}, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_2 {{1, 2, 64, 128}, {1, 2, 256, 64}, {1, 2, 256, 128}}, tensor{1}, tensor{0}, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_3 {{1, 1, 8, 16}, {1, 1, 32, 8}, {1, 1, 32, 16}}, tensor{1}, tensor{0}, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_U8U8_1 {{1, 1, 2, 2}, {1, 1, 2, 2}}, tensor{1}, tensor{0}, data_types::u8,  data_types::u8,  data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_2 {{1, 2, 64, 128}, {1, 2, 256, 64}}, tensor{1}, tensor{0}, data_types::u8,  data_types::u8,  data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_3 {{1, 1, 16, 32}, {1, 1, 12, 16}}, tensor{1}, tensor{0}, data_types::u8,  data_types::u8,  data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_U8S8_1 {{1, 1, 4, 2}, {1, 1, 8, 4}}, tensor{1}, tensor{0}, data_types::u8,  data_types::i8,  data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_S8U8_1 {{1, 2, 64, 128}, {1, 2, 256, 64}}, tensor{1}, tensor{0}, data_types::i8,  data_types::u8,  data_types::u8, format::bfyx, data_types::f32, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FP32 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
/* ----------- NOTE: A part of tests is disabled until all FP kernels don't support fusings ------------ */
class conv_fp32_activation : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::abs),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_activation, ::testing::ValuesIn(std::vector<bc_test_params>{
                                                                           bc_test_params{CASE_CONV_FP32_1, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP32_2, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP32_3, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP32_4, 2, 3},

                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
}), );


class conv_fp32_scale : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_scale, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_scale,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             // bc_test_params{CASE_CONV_FP32_1, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_2, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_10, 3, 3},

                                             // bc_test_params{CASE_CONV_FP16_1, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_2, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_10, 3, 3},
                                             }), );

class conv_fp32_prelu_eltwise : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_prelu_eltwise, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::bfyx_f16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops_mixed_types) {
    auto p = GetParam();
    auto slope_type = p.default_type == data_types::f32 ? data_types::f16 : data_types::f32;
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(layout{ slope_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} })),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::bfyx_f16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_prelu_eltwise,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             // bc_test_params{CASE_CONV_FP32_1, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_2, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 4},

                                             // bc_test_params{CASE_CONV_FP32_1, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_2, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 4},
                                             }), );

class conv_fp32_eltwise_bfzyx_f16 : public WeightsPrimitiveFusingTest {};

TEST_P(conv_fp32_eltwise_bfzyx_f16, vector_ops) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 eltwise("eltwise", "conv_prim", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::bfzyx_f16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_eltwise_bfzyx_f16, splitted_vector_ops) {
    auto p = GetParam();

    std::vector<std::string> weights_idx;
    for (size_t w = 0; w < p.groups; w++) {
        topology.add(data("weights" + std::to_string(w), get_mem(get_weights_layout(p, p.groups))));
        weights_idx.push_back(("weights" + std::to_string(w)));
    }

    topology.add(input_layout("input", get_input_layout(p)),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", weights_idx, {}, 1, p.stride, p.pad, p.dilation),
                 eltwise("eltwise", "conv_prim", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::bfzyx_f16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_eltwise_bfzyx_f16,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_6, 2, 3},
                                bc_test_params{CASE_CONV_FP32_7, 2, 3},
                                bc_test_params{CASE_CONV_FP32_8, 2, 3},
                                bc_test_params{CASE_CONV_FP32_9, 2, 3},

                                bc_test_params{CASE_CONV_FP16_6, 2, 3},
                                bc_test_params{CASE_CONV_FP16_7, 2, 3},
                                bc_test_params{CASE_CONV_FP16_8, 2, 3},
                                bc_test_params{CASE_CONV_FP16_9, 2, 3},
                        }), );

class conv_fp32_quantize_u8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_quantize_u8, DISABLED_basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 quantize("quantize", "conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_quantize_u8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_1, 2, 3},
                        }), );

class conv_fp32_scale_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_scale_quantize_i8, DISABLED_basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_scale_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_1, 2, 4},
                        }), );

class conv_fp32_scale_activation_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8, DISABLED_basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_1, 2, 5},
                        }), );

class conv_fp32_scale_activation_quantize_i8_eltwise_fp32 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_eltwise_fp32, DISABLED_basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum,  data_types::f32),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );
    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_eltwise_fp32,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_1, 2, 6},
                        }), );

class conv_fp32_scale_activation_quantize_i8_activation : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_activation, DISABLED_basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 activation("activation_quantize", "quantize", activation_func::relu),
                 reorder("reorder_bfyx", "activation_quantize", p.default_format, data_types::f32)
    );
    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_1, 2, 6},
                        }), );


class conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, DISABLED_basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(layout{data_types::i8, p.input_format, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_1, 2, 7},
                        }), );


/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------------- binary convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

class conv_bin_activation : public WeightsPrimitiveFusingTest {};
TEST_P(conv_bin_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 activation("activation", "bin_conv_prim", activation_func::relu),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 2, 3},
                                            }), );

class conv_bin_scale_activation : public WeightsPrimitiveFusingTest {};
TEST_P(conv_bin_scale_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 activation("activation", "scale", activation_func::relu),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_scale_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 2, 4},
                            bc_test_params{CASE_BIN_CONV2, 2, 4},
                                            }), );

class conv_bin_quantize_bin : public WeightsPrimitiveFusingTest {};
TEST_P(conv_bin_quantize_bin, channel_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("in_lo", in_thresh),
                 data("in_hi", in_thresh),
                 data("out_lo", get_mem(get_per_channel_layout(p), -1)),
                 data("out_hi", get_mem(get_per_channel_layout(p),  1)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 quantize("quantize_data", "bin_conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 2, data_types::bin),
                 reorder("reorder_bfyx", "quantize_data", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_quantize_bin, blob_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_single_element_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("in_lo", in_thresh),
                 data("in_hi", in_thresh),
                 data("out_lo", get_mem(get_single_element_layout(p), -1)),
                 data("out_hi", get_mem(get_single_element_layout(p), 1)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 quantize("quantize_data", "bin_conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 2, data_types::bin),
                 reorder("reorder_bfyx", "quantize_data", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_quantize_bin,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 2, 3},
                            bc_test_params{CASE_BIN_CONV2, 2, 3},
                                            }), );

class conv_bin_scale_conv_dw : public WeightsPrimitiveFusingTest {};
TEST_P(conv_bin_scale_conv_dw, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 2, 2};
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 reorder("reorder_bfyx", "conv_dw", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_scale_conv_dw, dw_kernel_3x3_stride1) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 1, 1};
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 reorder("reorder_bfyx", "conv_dw", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_scale_conv_dw,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV2, 3, 4},
                            bc_test_params{CASE_BIN_CONV3, 3, 4},
                                            }), );

class conv_bin_scale_conv_dw_prelu : public WeightsPrimitiveFusingTest {};
TEST_P(conv_bin_scale_conv_dw_prelu, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 2, 2};
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 activation("activation", "conv_dw", "slope_data", activation_func::relu_negative_slope),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_scale_conv_dw_prelu, dw_kernel_3x3_stride1) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 1, 1};
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 activation("activation", "conv_dw", "slope_data", activation_func::relu_negative_slope),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_scale_conv_dw_prelu,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV2, 3, 5},
                            bc_test_params{CASE_BIN_CONV3, 3, 5},
                                            }), );


/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- INT8 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class conv_int8_scale : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_scale, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_scale,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                        }), );

class conv_int8_prelu_eltwise : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_prelu_eltwise, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_prelu_eltwise,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 4},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 4},
                        }), );

class conv_int8_quantize_u8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_quantize_u8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 quantize("quantize", "conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_quantize_u8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                        }), );

class conv_int8_scale_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_scale_quantize_i8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_scale_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 4},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 4},
                        }), );

class conv_int8_scale_activation_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 5},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 5},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 5},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 5},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 5},
                        }), );

class conv_int8_scale_activation_quantize_i8_eltwise_fp32 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum,  data_types::f32),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );
    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 6},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 6},
                        }), );

class conv_int8_scale_activation_quantize_i8_activation : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 activation("activation_quantize", "quantize", activation_func::relu),
                 reorder("reorder_bfyx", "activation_quantize", p.default_format, data_types::f32)
    );
    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 6},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 6},
                        }), );


class conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(layout{data_types::i8, p.input_format, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 7},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 7},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 7},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 7},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 7},
                        }), );

class conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(layout{data_types::i8, format::b_fs_yx_fsv4, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops_mixed_types) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(layout{ data_types::f16, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} })),
                 data("eltwise_data", get_mem(layout{data_types::u8, format::b_fs_yx_fsv4, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_5, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_5, 2, 7},
                        }), );

class conv_int8_asymmetric_weights : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_asymmetric_weights, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("w_zp", get_mem(get_weights_zp_layout(p), 1, 127)),
                 eltwise("w_sub", {"weights", "w_zp"}, eltwise_mode::sub, data_types::f32),
                 convolution("conv_prim", "input", {"w_sub"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32),
                 reorder("reorder_bfyx", "conv_prim", p.default_format, data_types::f32)
    );
    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology, bo_not_fused);
    network network_fused(this->engine, this->topology, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 4lu);  // input + weights + bias + w_zp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_asymmetric_weights,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 2},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 2},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 2},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 2},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 2},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 2},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 2},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 2},
                        }), );

class conv_int8_asymmetric_data : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_asymmetric_data, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("a_zp", get_mem(get_activations_zp_layout(p), 1, 127)),
                 eltwise("a_sub", {"input", "a_zp"}, eltwise_mode::sub, data_types::f32),
                 convolution("conv_prim", "a_sub", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32),
                 reorder("reorder_bfyx", "conv_prim", p.default_format, data_types::f32)
    );
    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology, bo_not_fused);
    network network_fused(this->engine, this->topology, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 5lu);  // input + weights + bias + a_zp + comp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_asymmetric_data,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                        }), );

class conv_int8_asymmetric_data_and_weights : public WeightsPrimitiveFusingTest {};
TEST_P(conv_int8_asymmetric_data_and_weights, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("a_zp", get_mem(get_activations_zp_layout(p), 1, 127)),
                 data("w_zp", get_mem(get_weights_zp_layout(p), 1, 127)),
                 eltwise("a_sub", {"input", "a_zp"}, eltwise_mode::sub, data_types::f32),
                 eltwise("w_sub", {"weights", "w_zp"}, eltwise_mode::sub, data_types::f32),
                 convolution("conv_prim", "a_sub", {"w_sub"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32),
                 reorder("reorder_bfyx", "conv_prim", p.default_format, data_types::f32)
    );
    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology, bo_not_fused);
    network network_fused(this->engine, this->topology, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 6lu);  // input + weights + bias + a_zp + w_zp + comp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_int8_asymmetric_data_and_weights,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                        }), );

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FC cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class fc_fp32_activation : public WeightsPrimitiveFusingTest {};
TEST_P(fc_fp32_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                data("weights", get_mem(get_weights_layout(p))),
                data("bias", get_mem(get_bias_layout(p))),
                fully_connected("fc_prim", "input", "weights", "bias"),
                activation("activation", "fc_prim", activation_func::abs),
                reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, fc_fp32_activation, ::testing::ValuesIn(std::vector<bc_test_params>{
                                                                            bc_test_params{ CASE_FC_FP32_1, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_2, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_3, 2, 3 },
}), );

class fc_int8_scale : public WeightsPrimitiveFusingTest {};
TEST_P(fc_int8_scale, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count())),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32),
        scale("scale", "fc_prim", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, fc_int8_scale,
    ::testing::ValuesIn(std::vector<bc_test_params>{
                        bc_test_params{ CASE_FC_U8S8_1, 2, 3 },
                        bc_test_params{ CASE_FC_U8S8_2, 2, 3 },
                        bc_test_params{ CASE_FC_U8S8_3, 2, 3 },
                        }), );

class fc_int8_quantize_u8 : public WeightsPrimitiveFusingTest {};
TEST_P(fc_int8_quantize_u8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32),
        quantize("quantize", "fc_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu_fc, fc_int8_quantize_u8,
    ::testing::ValuesIn(std::vector<bc_test_params>{
        bc_test_params{CASE_FC_U8S8_1, 2, 3},
        bc_test_params{CASE_FC_U8S8_2, 2, 3},
        bc_test_params{CASE_FC_U8S8_3, 2, 3},
        }), );

class fc_int8_scale_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(fc_int8_scale_quantize_i8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32),
        scale("scale", "fc_prim", "scale_data"),
        quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, fc_int8_scale_quantize_i8,
    ::testing::ValuesIn(std::vector<bc_test_params>{
        bc_test_params{CASE_FC_U8S8_1, 2, 4},
        bc_test_params{CASE_FC_U8S8_2, 2, 4},
        bc_test_params{CASE_FC_U8S8_3, 2, 4},
        }), );



class fc_int8_scale_activation_quantize_i8 : public WeightsPrimitiveFusingTest {};
TEST_P(fc_int8_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32),
        scale("scale", "fc_prim", "scale_data"),
        activation("activation_scale", "scale", activation_func::exp),
        quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, fc_int8_scale_activation_quantize_i8,
    ::testing::ValuesIn(std::vector<bc_test_params>{
        bc_test_params{CASE_FC_U8S8_1, 2, 5},
        bc_test_params{CASE_FC_U8S8_2, 2, 5},
        bc_test_params{CASE_FC_U8S8_3, 2, 5},
        }), );

class gemm_int8_3in_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_int8_3in_quantize_i8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        input_layout("input2", get_input_layout(p, 2)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gemm("gemm_prim", { "input0", "input1", "input2" }, data_types::f32),
        quantize("quantize", "gemm_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, gemm_int8_3in_quantize_i8,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_3IN_S8S8_1, 4, 5 },
                        gemm_test_params{ CASE_GEMM_3IN_S8S8_2, 4, 5 },
                        gemm_test_params{ CASE_GEMM_3IN_S8S8_3, 4, 5 },
}), );

class gemm_int8_2in_quantize_u8 : public GemmFusingTest {};
TEST_P(gemm_int8_2in_quantize_u8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        quantize("quantize", "gemm_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, gemm_int8_2in_quantize_u8,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_1, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_2, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_3, 3, 4 },
}), );

class gemm_int8_2in_act_scale_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_int8_2in_act_scale_quantize_i8, basic) {
    auto p = GetParam();
    topology.add(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        activation("activation", "gemm_prim", activation_func::exp),
        scale("scale", "activation", "scale_data"),
        quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, gemm_int8_2in_act_scale_quantize_i8,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_2IN_U8S8_1, 3, 6 },
                        gemm_test_params{ CASE_GEMM_2IN_S8U8_1, 3, 6 },
}), );
