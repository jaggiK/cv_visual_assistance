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
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>

#include <api/data.hpp>
#include <api/reshape.hpp>
#include <api/input_layout.hpp>

#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;
using namespace testing;

void verify_float(const float& output_value, const float& value) {
    EXPECT_FLOAT_EQ(output_value, value);
}

void verify_int(const int32_t& output_value, const int32_t& value) {
    EXPECT_EQ(output_value, value);
}

template <class ElemType>
void generic_reshape_test(format fmt, tensor const& input_size, tensor const& reshape_size, bool in_place, padding const& input_padd = padding(), padding const& output_padd = padding()) {
    const auto& engine = get_test_engine();

    //allocate input memory
    auto data_type = data_types::f32;
    if (std::is_same<ElemType, FLOAT16>::value)
        data_type = data_types::f16;
    else if (std::is_same<ElemType, int8_t>::value)
        data_type = data_types::i8;
    else if (std::is_same<ElemType, int32_t>::value)
        data_type = data_types::i32;
    else if (std::is_same<ElemType, int64_t>::value)
        data_type = data_types::i64;

    auto input = memory::allocate(engine, {data_type, fmt, input_size});

    {
        auto input_ptr = input.cldnn::memory::pointer<ElemType>();
        auto input_itr = input_ptr.begin();

        auto elements = input_size.count();

        int value = 1;
        for (size_t i = 0; i < elements; ++i)
            *input_itr++ = (ElemType)value++;
    }

    topology tpl;
    std::string reshape_input = "input";

    tpl.add(input_layout("input", input.get_layout()));
    if (input_padd) {
        auto padded_input_layout = input.get_layout();
        padded_input_layout.data_padding = input_padd;
        tpl.add(reorder("reorder", "input", padded_input_layout));
        reshape_input = "reorder";
    }
    tpl.add(reshape("reshape", reshape_input, reshape_size, output_padd));

    build_options bo;
    bo.set_option(build_option::outputs({reshape_input, "reshape"}));

    network net(engine, tpl, bo);
    net.set_input_data("input", input);
    auto outputs = net.execute();

    ASSERT_TRUE(outputs.size() == 2 && outputs.count("reshape") == 1 && outputs.count(reshape_input) == 1);
    auto net_input = outputs.at(reshape_input).get_memory();
    auto output = outputs.at("reshape").get_memory();

    EXPECT_TRUE(output.get_layout().data_type == input.get_layout().data_type);        //reshape should not change data_type
    EXPECT_TRUE(output.get_layout().format.value == input.get_layout().format.value);  //reshape should not change format

    //output size should be equal to requested plus output padding
    ASSERT_TRUE(output.get_layout().size == reshape_size);
    ASSERT_TRUE(output.get_layout().get_buffer_size() == reshape_size.add(output_padd.lower_size()).add(output_padd.upper_size()));

    {
        auto output_ptr = output.pointer<const ElemType>();
        auto output_itr = output_ptr.begin();

        auto sizes = reshape_size.sizes(fmt);
        auto lower = output_padd.lower_size().sizes(fmt);
        auto upper = output_padd.upper_size().sizes(fmt);
        auto buffer_sizes = sizes;
        int32_t accum = 1;
        for (size_t i = 1; i <= sizes.size(); ++i) {
            buffer_sizes[sizes.size() - i] = accum;
            accum *= lower[sizes.size() - i] + sizes[sizes.size() - i] + upper[sizes.size() - i];
        }

        int value = 1;

        output_itr += lower[0] * buffer_sizes[0];
        for (int d1 = 0; d1 < sizes[0]; ++d1) {
            output_itr += lower[1] * buffer_sizes[1];
            for (int d2 = 0; d2 < sizes[1]; ++d2) {
                output_itr += lower[2] * buffer_sizes[2];
                for (int d3 = 0; d3 < sizes[2]; ++d3) {
                    output_itr += lower[3] * buffer_sizes[3];
                    for (int d4 = 0; d4 < sizes[3]; ++d4) {
                        auto& output_value = *output_itr;
                        ++output_itr;
                        if (data_type == data_types::f16 || data_type == data_types::f32)
                            verify_float(static_cast<float>(output_value), static_cast<float>((ElemType)value));
                        else
                            verify_int(static_cast<int32_t>(output_value), static_cast<int32_t>(value));
                        ++value;
                    }

                    output_itr += upper[3] * buffer_sizes[3];
                }

                output_itr += upper[2] * buffer_sizes[2];
            }

            output_itr += upper[1] * buffer_sizes[1];
        }
    }
}

TEST(reshape_gpu_f32, basic_2dim_in_place) {
    generic_reshape_test<float>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 4, 1),
        true);
}

TEST(reshape_gpu_f16, basic_2dim_in_place) {
    generic_reshape_test<FLOAT16>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_i8, basic_2dim_in_place) {
    generic_reshape_test<int8_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_i32, basic_2dim_in_place) {
    generic_reshape_test<int32_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_i64, basic_2dim_in_place) {
    generic_reshape_test<int64_t>(
        format::bfyx,
        tensor(1, 1, 2, 2),
        tensor(1, 1, 1, 4),
        true);
}

TEST(reshape_gpu_f32, basic_4dim_in_place) {
    generic_reshape_test<float>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(27, 2, 3, 4),
        true);
}

TEST(reshape_gpu_f16, basic_4dim_in_place) {
    generic_reshape_test<FLOAT16>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true);
}

TEST(reshape_gpu_i32, basic_4dim_in_place) {
    generic_reshape_test<int32_t>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true);
}

TEST(reshape_gpu_i64, basic_4dim_in_place) {
    generic_reshape_test<int64_t>(
        format::yxfb,
        tensor(9, 9, 2, 4),
        tensor(3, 4, 27, 2),
        true);
}

TEST(reshpape_gpu_f32, basic_2dim_output_padd) {
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 4, 2),
        tensor(1, 1, 8, 1),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 1, 1}));
}

TEST(reshape_gpu_f16, basic_2dim_output_padd) {
    generic_reshape_test<FLOAT16>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_i8, basic_2dim_output_padd) {
    generic_reshape_test<int8_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_i32, basic_2dim_output_padd) {
    generic_reshape_test<int32_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_i64, basic_2dim_output_padd) {
    generic_reshape_test<int64_t>(
        format::byxf,
        tensor(1, 1, 3, 4),
        tensor(1, 1, 2, 6),
        false,
        padding(),
        padding(std::vector<int>{0, 0, 2, 2}));
}

TEST(reshape_gpu_f32, basic_2dim_input_padd) {
    generic_reshape_test<float>(
        format::fyxb,
        tensor(1, 1, 2, 5),
        tensor(1, 1, 5, 2),
        false,
        padding({0, 0, 3, 2}, {0, 0, 1, 4}));
}

TEST(reshape_gpu_f16, basic_2dim_input_padd) {
    generic_reshape_test<FLOAT16>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_i8, basic_2dim_input_padd) {
    generic_reshape_test<int8_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_i32, basic_2dim_input_padd) {
    generic_reshape_test<int32_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_i64, basic_2dim_input_padd) {
    generic_reshape_test<int64_t>(
        format::fyxb,
        tensor(1, 1, 3, 3),
        tensor(1, 1, 1, 9),
        false,
        padding({0, 0, 4, 1}, {0, 0, 2, 3}));
}

TEST(reshape_gpu_f32, basic_2dim_input_output_padd) {
    generic_reshape_test<float>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshape_gpu_f16, basic_2dim_input_output_padd) {
    generic_reshape_test<FLOAT16>(
        format::byxf,
        tensor(1, 1, 6, 6),
        tensor(1, 1, 3, 12),
        false,
        padding({0, 0, 1, 1}, {0, 0, 0, 0}),
        padding({0, 0, 2, 1}, {0, 0, 1, 2}));
}

TEST(reshape_gpu_i8, basic_2dim_input_output_padd) {
    generic_reshape_test<int8_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshape_gpu_i32, basic_2dim_input_output_padd) {
    generic_reshape_test<int32_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshape_gpu_i64, basic_2dim_input_output_padd) {
    generic_reshape_test<int64_t>(
        format::byxf,
        tensor(1, 1, 5, 7),
        tensor(1, 1, 7, 5),
        false,
        padding({0, 0, 4, 4}, {0, 0, 1, 1}),
        padding({0, 0, 0, 0}, {0, 0, 3, 0}));
}

TEST(reshpape_gpu_f32, basic_4dim_output_padd) {
    generic_reshape_test<float>(
        format::bfyx,
        tensor(2, 5, 7, 3),
        tensor(1, 14, 15, 1),
        false,
        padding(),
        padding({1, 0, 0, 1}, {0, 2, 3, 0}));
}

TEST(reshape_gpu_f16, basic_4dim_output_padd) {
    generic_reshape_test<FLOAT16>(
        format::bfyx,
        tensor(5, 4, 2, 2),
        tensor(40, 2, 1, 1),
        false,
        padding(),
        padding({0, 2, 0, 1}, {0, 2, 3, 0}));
}

TEST(reshape_gpu_f32, basic_4dim_input_padd) {
    generic_reshape_test<float>(
        format::yxfb,
        tensor(8, 128, 3, 3),
        tensor(16, 8, 8, 9),
        false,
        padding({0, 1, 3, 3}, {0, 1, 1, 1}));
}

TEST(reshape_gpu_f16, basic_4dim_input_padd) {
    generic_reshape_test<FLOAT16>(
        format::yxfb,
        tensor(2, 32, 8, 8),
        tensor(8, 128, 1, 4),
        false,
        padding({2, 2, 1, 0}, {1, 2, 2, 0}));
}

TEST(reshape_gpu_f32, basic_4dim_input_output_padd) {
    generic_reshape_test<float>(
        format::fyxb,
        tensor(8, 1024, 25, 25),
        tensor(8, 64, 100, 100),
        false,
        padding({2, 0, 2, 1}, {0, 1, 4, 0}),
        padding({1, 2, 3, 4}, {0, 4, 1, 1}));
}

TEST(reshape_gpu_f16, basic_4dim_input_output_padd) {
    generic_reshape_test<FLOAT16>(
        format::byxf,
        tensor(32, 3, 227, 227),
        tensor(8, 12, 227, 227),
        false,
        padding({0, 1, 4, 4}, {0, 1, 1, 1}),
        padding({0, 29, 29, 0}, {0, 0, 0, 0}));
}

TEST(reshape_gpu_f32, basic_5dim_in_place) {
    generic_reshape_test<float>(
        format::bfzyx,
        tensor(9, 9, 2, 4, 2),
        tensor(27, 2, 1, 4, 6),
        true);
}

TEST(reshape_gpu_f32, multiple_users_with_reorder) {
    // Tests split with crop implementation
    //                                                   _ REORDER(yxfb) --> RELU(yxfb)
    //                                                  |
    //  INPUT(bfyx,2x2x1x1)--RELU(bfyx)--RESHAPE(4x1x1x1)
    //                                                  |_
    //                                                     RELU(bfyx)

    //  Input:
    //  b0f0: -1.0
    //  b0f1:  2.0
    //  b1f0: -3.0
    //  b1f1:  4.0

    //  Out1:
    //  b0f0:  0.0
    //  b0f1:  0.0
    //  b1f0:  2.0
    //  b1f1:  4.0

    //  Out2:
    //  b0f0:  0.0
    //  b0f1:  2.0
    //  b1f0:  0.0
    //  b1f1:  4.0

    const auto& engine = get_test_engine();
    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 1;
    auto y_size = 1;
    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num))}});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(activation("relu", "input", activation_func::relu));
    topology.add(reshape("reshape", "relu", tensor(batch(4))));
    topology.add(reorder("reorder1", "reshape", format::yxfb, data_types::f32));
    topology.add(activation("relu1", "reorder1", activation_func::relu));
    topology.add(activation("relu2", "reshape", activation_func::relu));

    std::vector<float> input_vec = {-1.f, 2.f, -3.f, 4.f};
    std::vector<float> out1 = {0.f, 0.f, 2.f, 4.0f};
    std::vector<float> out2 = {0.f, 2.f, 0.f, 4.0f};
    set_values(input, input_vec);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("relu1").get_memory();
    auto output_ptr = output.pointer<float>();

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("relu2").get_memory();
    auto output_ptr_2 = output_2.pointer<float>();

    for (size_t i = 0; i < out2.size(); i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(reshape_gpu_f32, calc_output_shape) {
    //  INPUT(bfyx,2x2x1x1) -- RESHAPE(1, 1, 0, -1)

    //  Input:
    //  b0f0: -1.0
    //  b0f1:  2.0
    //  b1f0: -3.0
    //  b1f1:  4.0
    //
    // output_shape (1, 1, 1, 4)

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, {data_types::f32, format::bfyx, {2, 2, 1, 1}});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reshape("reshape", "input", tensor(1, 1, 0, -1)));

    set_values(input, {-1.f, 2.f, -3.f, 4.f});

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reshape");

    auto output = outputs.at("reshape").get_memory();

    EXPECT_TRUE(output.get_layout().data_type == input.get_layout().data_type);
    EXPECT_TRUE(output.get_layout().format == input.get_layout().format);

    ASSERT_TRUE(output.get_layout().size == tensor(1, 1, 1, 4));

    float answers[4] = {-1.f, 2.f, -3.f, 4.f};

    auto output_ptr = output.pointer<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}

TEST(reshape_gpu_f32, basic_bfwzyx) {
    // input:  bfwzyx, (3, 3, 2, 2, 1, 1)
    // reshape: (1, 1, 2, 2, 3, 3), pad (0, 0, 0, 0, 0, 1)

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, layout{data_types::f32, format::bfwzyx, tensor{batch(3), feature(3), spatial(1, 1, 2, 2)}});

    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(reshape("reshape", "input", tensor(batch(1), feature(1), spatial(2, 2, 3, 3)), padding({0, 0, 0, 0, 0, 1}, 0.f)));

    // clang-format off
    std::vector<float> input_data = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
    };

    std::vector<float> expected_out = {
        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,

        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 1.f, 2.f, 3.f,

        4.f, 5.f, 6.f, 7.f,
        8.f, 9.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 6.f,

        7.f, 8.f, 9.f, 1.f,
        2.f, 3.f, 4.f, 5.f,
        6.f, 7.f, 8.f, 9.f,

        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f,
    };
    // clang-format on

    set_values(input, input_data);

    network network(engine, topology);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "reshape");

    auto output = outputs.at("reshape").get_memory();

    EXPECT_TRUE(output.get_layout().data_type == input.get_layout().data_type);
    EXPECT_TRUE(output.get_layout().format == input.get_layout().format);

    auto output_ptr = output.pointer<float>();
    ASSERT_EQ(output_ptr.size(), expected_out.size());

    for (size_t i = 0; i < expected_out.size(); i++) {
        EXPECT_TRUE(are_equal(expected_out[i], output_ptr[i]));
    }
}
