/*
// Copyright (c) 2017-2019 Intel Corporation
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
#include <api/input_layout.hpp>
#include "api/crop.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

template<typename T>
std::vector<T> generate_random_input(size_t b, size_t f, size_t z, size_t y, size_t x, int min, int max) {
    static std::default_random_engine generator(random_seed);
    int k = 8; // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int> distribution(k * min, k * max);
    std::vector<T> v(b*f*x*y*z);
    for (size_t i = 0; i < b*f*x*y*z; ++i) {
        v[i] = (T)distribution(generator);
        v[i] /= k;
    }
    return v;
}

template<typename T>
std::vector<T> generate_random_input(size_t b, size_t f, size_t y, size_t x, int min, int max) {
    return generate_random_input<T>(b, f, 1, y, x, -min, max);
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x2x2x3_crop_all) {
    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<float> input_vec;
    for (int i = 0; i < batch_num * feature_num * y_size * x_size; i++)
        input_vec.push_back(static_cast<float>(i));
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();

    printf("Results:\n");
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_int_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = memory::allocate(engine, { data_types::i32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = generate_random_input<int32_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<int32_t>();

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    const auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();
    std::vector<float> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_int_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    const auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = memory::allocate(engine, { data_types::i32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = generate_random_input<int32_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<int32_t>();
    std::vector<int32_t> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    const auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = memory::allocate(engine, { data_types::f32,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_int_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    const auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = memory::allocate(engine, { data_types::i32,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = generate_random_input<int32_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<int32_t>();
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_int_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   50   -5   -15
    //  f1: b0:  5    6  -12   b1:   15   52   -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    const auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = memory::allocate(engine, { data_types::i32, format::yxfb,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<int32_t> input_vec = { 1, 0, 5, 15,
        2, 0, 6, 52,
        -10, -11, -12, -13,
        3, 50, 7, 12,
        4, -5, 8, 8,
        -14, -15, -16, -17 };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<int32_t>();

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0
    //  f3:  4.0

    //  Out1:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0

    //  Out2:
    //  f0: 4.0
    const auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop1", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { -1.f, 2.f,-3.f };
    std::vector<float> out2 = { 4.f, };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::outputs(topology.get_primitive_ids()));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < out1.size();i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    std::cout << std::endl;
    auto output_2 = outputs.at("crop2").get_memory();
    auto output_ptr_2 = output_2.pointer<float>();

    for (size_t i = 0; i < out2.size();i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_int_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1
    //  f1:  2
    //  f2: -3
    //  f3:  4

    //  Out1:
    //  f0: -1
    //  f1:  2
    //  f2: -3

    //  Out2:
    //  f0: 4
    const auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = memory::allocate(engine, { data_types::i32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop1", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<int32_t> input_vec = { -1, 2, -3, 4 };
    std::vector<int32_t> out1 = { -1, 2,-3 };
    std::vector<int32_t> out2 = { 4, };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::outputs(topology.get_primitive_ids()));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    auto output_ptr = output.pointer<int32_t>();

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    std::cout << std::endl;
    auto output_2 = outputs.at("crop2").get_memory();
    auto output_ptr_2 = output_2.pointer<int32_t>();

    for (size_t i = 0; i < out2.size(); i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in1x4x1x1_split_w_relu) {
    // Tests split with crop implementation
    //                        _ CROP_1(1x3x1x1,offset(0x0x0x0)) --> RELU
    //                       |
    //  INPUT(1x4x1x1)--RELU
    //                       |_
    //                          CROP_2(1x1x1x1,offset(0x3x0x0)) --> RELU
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0
    //  f3:  4.0

    //  Out1:
    //  f0: 0.0
    //  f1: 2.0
    //  f2: 0.0

    //  Out2:
    //  f0: 4.0
    // disable memory pool when we want to check optimized out internal results
    engine_configuration cfg{ false, false, false, std::string(), std::string(), true /*oooq*/, std::string(),std::string(), priority_mode_types::disabled,  throttle_mode_types::disabled, false /*mem_pool*/ };
    engine engine{ cfg };
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;
    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_func::relu));
    topology.add(crop("crop1", "relu", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "relu", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));
    topology.add(activation("relu1", "crop1", activation_func::relu));
    topology.add(activation("relu2", "crop2", activation_func::relu));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { 0.f, 2.f,0.f };
    std::vector<float> out2 = { 4.f, };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::debug(true)); //required to have optimized crop despite the fact that it's specified as an output

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("relu1").get_memory();
    auto output_ptr = output.pointer<float>();

    // check if crop has been executed in place
    auto in_place = outputs.at("crop1").get_memory().is_the_same_buffer(outputs.at("relu").get_memory());
    EXPECT_TRUE(in_place);

    for (size_t i = 0; i < out1.size();i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("relu2").get_memory();
    auto output_ptr_2 = output_2.pointer<float>();

    for (size_t i = 0; i < out2.size();i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in3x1x2x2x1_crop_all_bfzyx) {
    //  Reference  : 3x1x2x2x1
    //  Input      : 6x2x4x3x2
    //  Output     : 3x1x2x2x1

    const auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;
    auto z_size = 2;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;
    auto crop_z_size = z_size - 1;

    auto input = memory::allocate(engine, { data_types::f32,format::bfzyx,{ batch_num, feature_num, x_size, y_size, z_size } });

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size, crop_z_size }, { 0, 0, 0, 0, 0 }));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    auto output_ptr = output.pointer<float>();
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int z = 0; z < crop_z_size; ++z) { //Z
                for (int y = 0; y < crop_y_size; ++y) { //Y
                    for (int x = 0; x < crop_x_size; ++x) { //X
                        int linear_id = x + x_size * (y + y_size * (z + z_size * (f + feature_num * b)));
                        int output_linear_id = x + crop_x_size * (y + crop_y_size * (z + crop_z_size * (f + crop_feature_num * b)));
                        EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                    }
                }
            }
        }
    }
}
