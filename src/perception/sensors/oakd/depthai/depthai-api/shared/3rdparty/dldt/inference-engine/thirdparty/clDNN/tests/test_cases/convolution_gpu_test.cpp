﻿/*
// Copyright (c) 2016-2019 Intel Corporation
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
#include <gmock/gmock.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/convolution.hpp"
#include "api/eltwise.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include "test_utils/float16.h"
#include <api/data.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <type_traits>
#include <fstream>
#include <tuple>
#include <api/reorder.hpp>
#include <src/include/to_string_utils.h>

using namespace cldnn;
using namespace tests;

namespace cldnn
{
    template<> struct type_to_data_type<FLOAT16> { static const data_types value = data_types::f16; };
}

template<typename T>
T kahan_summation(std::vector<T> &input) {
    T sum = 0;
    T c = 0;
    for (T x : input) {
        T y = x - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template <typename InputT>
struct convolution_accumulator {
    using type = InputT;
};

template <>
struct convolution_accumulator<int8_t> {
    using type = int;
};

template <>
struct convolution_accumulator<uint8_t> {
    using type = int;
};

template<typename InputT, typename OutputT = InputT, typename WeightsT = InputT,  typename AccT = typename convolution_accumulator<InputT>::type>
VVF<OutputT> reference_convolve(VVVF<InputT> &input, VVVF<WeightsT> &filter, int stride_y, int stride_x, float bias, int dilation_y = 1, int dilation_x = 1,
        int input_padding_y = 0, int input_padding_x = 0, int output_padding_y = 0,
        int output_padding_x = 0, size_t f_begin = 0, size_t f_end = 0, bool depthwise = false,
        const VF<InputT>& data_zp = {}, const WeightsT& weights_zp = 0)
{
    size_t kernel_extent_y = dilation_y * (filter[0].size() - 1) + 1;
    size_t kernel_extent_x = dilation_x * (filter[0][0].size() - 1) + 1;
    size_t output_y = 1 + (input[0].size() - kernel_extent_y + 2 * input_padding_y) / stride_y + 2 * output_padding_y;
    size_t output_x = 1 + (input[0][0].size() - kernel_extent_x + 2 * input_padding_x) / stride_x + 2 * output_padding_x;
    bool asymm_data = !data_zp.empty();
    bool asymm_weights = weights_zp != static_cast<WeightsT>(0);
    VVF<OutputT> output(output_y, VF<OutputT>(output_x, 0));
    size_t filter_begin = f_begin ? f_begin : 0;
    size_t filter_end = f_end ? f_end : filter.size();
    for (size_t f = filter_begin; f < filter_end; ++f) {
        for (size_t y = 0; y < (output_y - 2 * output_padding_y); ++y) {
            for (size_t x = 0; x < (output_x - 2 * output_padding_x); ++x) {
                VF<AccT> values;
                values.reserve(filter[0].size() * filter[0][0].size());
                for (size_t yf = 0; yf < filter[0].size(); ++yf) {
                    int yi = -input_padding_y + (int)yf * dilation_y + stride_y * (int)y;
                    bool yi_inside = yi >= 0 && (int)input[0].size() > yi;
                    if (!yi_inside && !asymm_data) continue;
                    for (size_t xf = 0; xf < filter[0][0].size(); ++xf) {
                        int xi = -input_padding_x + (int)xf * dilation_x + stride_x * (int)x;
                        bool xi_inside = xi >= 0 && (int)input[0][0].size() > xi;
                        if (!xi_inside && !asymm_data) continue;

                        AccT input_val;
                        if (xi_inside && yi_inside) {
                            input_val = static_cast<AccT>(input[f][yi][xi]);
                        } else {
                            input_val = static_cast<AccT>(0);
                        }

                        if (asymm_data) {
                            input_val = input_val - static_cast<AccT>(data_zp[f]);
                        }

                        AccT weights_val;
                        if (!depthwise) {
                            weights_val = static_cast<AccT>(filter[f][yf][xf]);
                        } else {
                            weights_val = static_cast<AccT>(filter[0][yf][xf]);
                        }

                        if (asymm_weights) {
                            weights_val = weights_val - static_cast<AccT>(weights_zp);
                        }

                        values.push_back(input_val * weights_val);
                    }
                }
                output[y + output_padding_y][x + output_padding_x] += static_cast<OutputT>(kahan_summation<AccT>(values));
            }
        }
    }

    for (size_t y = 0; y < (output_y - 2 * output_padding_y); ++y) {
        for (size_t x = 0; x < (output_x - 2 * output_padding_x); ++x) {
            output[y + output_padding_y][x + output_padding_x] += static_cast<OutputT>(bias);
        }
    }
    return output;
}

template <typename T>
VVF<T> reference_scale_post_op(const VVF<T>& input, const T& scale, const T& shift) {
    auto output = input;
    auto size_y = input.size();
    auto size_x = input[0].size();
    for (size_t yi = 0; yi < size_y; ++yi) {
        for (size_t xi = 0; xi < size_x; ++xi) {
            output[yi][xi] = output[yi][xi] * scale + shift;
        }
    }
    return output;
}

void dump_buffer(memory const& mem, std::string const& name)
{
    std::ofstream out(name);
    auto size = mem.get_layout().get_buffer_size();
    auto ptr = mem.pointer<const float>();
    auto pitches = mem.get_layout().get_pitches();
    out << "Data size: " << mem.get_layout().size << "\n";
    out << "Lower padding: " << mem.get_layout().data_padding.lower_size() << "\n";
    out << "Upper padding: " << mem.get_layout().data_padding.upper_size() << "\n";
    out << "\n";

    for (int b = 0; b < size.batch[0]; ++b)
    {
        out << " ================ BATCH " << b << " =================\n\n";
        for (int f = 0; f < size.feature[0]; ++f)
        {
            out << "feature " << f << ":\n";
            for (int y = 0; y < size.spatial[1]; ++y)
            {
                for (int x = 0; x < size.spatial[0]; ++x)
                {
                    size_t idx = b * pitches.batch[0] + f * pitches.feature[0] + y * pitches.spatial[1] + x * pitches.spatial[0];
                    out << ptr[idx] << " ";
                }
                out << "\n";
            }

            out << "\n";
        }

        out << "\n";
    }
}

TEST(deformable_convolution_f32_fw_gpu, basic_deformable_convolution_def_group1_2) {
    //  Input    : 4x4x4
    //  Trans    : 18x4x4
    //  Output   : 4x4x4
    //  In_offset: 1x1
    //  Filter   : 3x3
    //  Stride   : 1x1
    //  Dilation : 1x1
    //  Group    : 1
    //  Def_group: 1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 4, 4 } });
    auto trans = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 18, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 4, 4, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } });

    set_values(input, { 0.680375f, -0.211234f, 0.566198f, 0.59688f, 0.823295f, -0.604897f, -0.329554f, 0.536459f,
                        -0.444451f, 0.10794f, -0.0452059f, 0.257742f, -0.270431f, 0.0268018f, 0.904459f, 0.83239f,
                        0.271423f, 0.434594f, -0.716795f, 0.213938f, -0.967399f, -0.514226f, -0.725537f, 0.608353f,
                        -0.686642f, -0.198111f, -0.740419f, -0.782382f, 0.997849f, -0.563486f, 0.0258648f, 0.678224f,
                        0.22528f, -0.407937f, 0.275105f, 0.0485743f, -0.012834f, 0.94555f, -0.414966f, 0.542715f,
                        0.0534899f, 0.539828f, -0.199543f, 0.783059f, -0.433371f, -0.295083f, 0.615449f, 0.838053f,
                        -0.860489f, 0.898654f, 0.0519907f, -0.827888f, -0.615572f, 0.326454f, 0.780465f, -0.302214f,
                        -0.871657f, -0.959954f, -0.0845965f, -0.873808f, -0.52344f, 0.941268f, 0.804416f, 0.70184f });

    set_values(trans, { -0.466668f, 0.0795207f, -0.249586f, 0.520497f, 0.0250708f, 0.335448f, 0.0632129f, -0.921439f, -0.124725f,
                        0.86367f, 0.86162f, 0.441905f, -0.431413f, 0.477069f, 0.279958f, -0.291903f, 0.375723f, -0.668052f,
                        -0.119791f, 0.76015f, 0.658402f, -0.339326f, -0.542064f, 0.786745f, -0.29928f, 0.37334f, 0.912936f,
                        0.17728f, 0.314608f, 0.717353f, -0.12088f, 0.84794f, -0.203127f, 0.629534f, 0.368437f, 0.821944f,
                        -0.0350187f, -0.56835f, 0.900505f, 0.840256f, -0.70468f, 0.762124f, 0.282161f, -0.136093f, 0.239193f,
                        -0.437881f, 0.572004f, -0.385084f, -0.105933f, -0.547787f, -0.624934f, -0.447531f, 0.112888f, -0.166997f,
                        -0.660786f, 0.813608f, -0.793658f, -0.747849f, -0.00911188f, 0.52095f, 0.969503f, 0.870008f, 0.36889f,
                        -0.233623f, 0.499542f, -0.262673f, -0.411679f, -0.535477f, 0.168977f, -0.511175f, -0.69522f, 0.464297f,
                        -0.74905f, 0.586941f, -0.671796f, 0.490143f, -0.85094f, 0.900208f, -0.894941f, 0.0431267f, -0.647579f,
                        -0.519875f, 0.595596f, 0.465309f, 0.313127f, 0.93481f, 0.278917f, 0.51947f, -0.813039f, -0.730195f,
                        0.0404202f, -0.843536f, -0.860187f, -0.59069f, -0.077159f, 0.639355f, 0.146637f, 0.511162f, -0.896122f,
                        -0.684386f, 0.999987f, -0.591343f, 0.779911f, -0.749063f, 0.995598f, -0.891885f, 0.74108f, -0.855342f,
                        -0.991677f, 0.846138f, 0.187784f, -0.639255f, -0.673737f, -0.21662f, 0.826053f, 0.63939f, -0.281809f,
                        0.10497f, 0.15886f, -0.0948483f, 0.374775f, -0.80072f, 0.0616159f, 0.514588f, -0.39141f, 0.984457f,
                        0.153942f, 0.755228f, 0.495619f, 0.25782f, -0.929158f, 0.495606f, 0.666477f, 0.850753f, 0.746543f,
                        0.662075f, 0.958868f, 0.487622f, 0.806733f, 0.967191f, 0.333761f, -0.00548297f, -0.672064f, 0.660024f,
                        0.777897f, -0.846011f, 0.299414f, -0.503912f, 0.258959f, -0.541726f, 0.40124f, -0.366266f, -0.342446f,
                        -0.537144f, -0.851678f, 0.266144f, -0.552687f, 0.302264f, 0.021372f, 0.942931f, -0.439916f, 0.0922137f,
                        0.438537f, -0.773439f, -0.0570331f, 0.18508f, 0.888636f, -0.0981649f, -0.327298f, 0.695369f, -0.130973f,
                        -0.993537f, -0.310114f, 0.196963f, 0.666487f, -0.532217f, 0.350952f, -0.0340995f, -0.0361283f, -0.390089f,
                        0.424175f, -0.634888f, 0.243646f, -0.918271f, -0.172033f, 0.391968f, 0.347873f, 0.27528f, -0.305768f,
                        -0.630755f, 0.218212f, 0.254316f, 0.461459f, -0.343251f, 0.480877f, -0.595574f, 0.841829f, 0.369513f,
                        0.306261f, -0.485469f, 0.0648819f, -0.824713f, -0.479006f, 0.754768f, 0.37225f, -0.81252f, -0.777449f,
                        -0.276798f, 0.153381f, 0.186423f, 0.333113f, -0.422444f, 0.551535f, -0.423241f, -0.340716f, -0.620498f,
                        0.968726f, -0.992843f, 0.654782f, -0.337042f, -0.623598f, -0.127006f, 0.917274f, 0.837861f, 0.529743f,
                        0.398151f, -0.757714f, 0.371572f, -0.232336f, 0.548547f, 0.886103f, 0.832546f, 0.723834f, -0.592904f,
                        0.587314f, 0.0960841f, -0.405423f, 0.809865f, 0.819286f, 0.747958f, -0.00371218f, 0.152399f, -0.674487f,
                        -0.452178f, 0.729158f, -0.0152023f, -0.0726757f, 0.697884f, -0.0080452f, -0.417893f, -0.639158f, 0.368357f,
                        0.455101f, -0.721884f, 0.206218f, -0.0151566f, 0.676267f, 0.448504f, -0.643585f, -0.556069f, -0.00294906f,
                        -0.757482f, -0.723523f, -0.279115f, -0.350386f, 0.863791f, 0.816969f, 0.244191f, 0.673656f, 0.636255f,
                        -0.00785118f, -0.330057f, -0.211346f, 0.317662f, 0.217766f, -0.482188f, -0.69754f, -0.85491f, -0.784303f,
                        0.294415f, -0.272803f, -0.423461f, -0.337228f, -0.817703f, -0.145345f, 0.868989f, 0.167141f, -0.469077f });

    set_values(weights, { 0.0f, 0.841471f, 0.909297f, 0.14112f, -0.756802f, -0.958924f, -0.279415f, 0.656987f, 0.989358f,
                          0.412118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.912945f, 0.836656f,
                          -0.00885131f, -0.84622f, -0.905578f, -0.132352f, 0.762558f, 0.956376f, 0.270906f, -0.663634f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.745113f, -0.158623f, -0.916522f,
                          -0.831775f, 0.0177019f, 0.850904f, 0.901788f, 0.123573f, -0.768255f, -0.953753f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.304811f, -0.966118f, -0.739181f, 0.167356f,
                          0.920026f, 0.826829f, -0.0265512f, -0.85552f, -0.897928f, -0.114785f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.993889f, -0.629888f, 0.313229f, 0.968364f, 0.73319f,
                          -0.176076f, -0.923458f, -0.821818f, 0.0353983f, 0.860069f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, -0.506366f, 0.452026f, 0.994827f, 0.622989f, -0.321622f, -0.970535f,
                          -0.727143f, 0.184782f, 0.926818f, 0.816743f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.580611f, 0.998815f, 0.498713f, -0.459903f, -0.995687f, -0.61604f, 0.329991f,
                          0.97263f, 0.721038f, -0.193473f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.98024f, 0.363171f, -0.587795f, -0.998345f });

    set_values(biases, { -0.491022f, 0.467745f, 0.996469f, 0.609044f });

    std::vector<float> output_vec = {
            -0.0742483f, -2.09984f, -0.850381f, 0.0398813f, -1.06922f, -0.0233979f, -1.15264f, -0.970688f, -0.0428347f,
            -1.73668f, 0.613717f, 2.12469f, 0.450835f, -1.82602f, -0.57416f, -0.682909f, -0.211437f, 0.351543f,
            0.930845f, 0.412505f, 1.23318f, 0.894126f, 1.56587f, 0.882479f, 0.640997f, -1.94229f, -1.00846f, -3.64185f,
            -0.383791f, -0.274041f, -1.49479f, -2.82027f, 0.858848f, 1.7228f, 0.51184f, 0.537693f, 1.40331f, 0.192823f,
            0.325383f, 0.814044f, 1.19015f, 0.403436f, 1.40995f, 0.42931f, 0.131369f, 2.01262f, 0.253117f, 0.018361f,
            1.3469f, 1.15957f, 0.599044f, 1.48224f, 2.16468f, 0.504246f, -1.52044f, 0.10271f, 0.0379517f, 0.942841f,
            -2.6067f, 0.562893f, 0.671884f, 0.404735f, 1.45044f, 0.950113f };

    topology topology(
            input_layout("input", input.get_layout()),
            input_layout("trans", trans.get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution(
                    "conv",
                    "input",
                    "trans",
                    { "weights" },
                    { "biases" },
                    1,
                    1,
                    { 1, 1, 1, 1 },
                    { 0, 0, -1, -1 },
                    { 1, 1, 1, 1 },
                    { 1, 4, 4, 4 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);
    network.set_input_data("trans", trans);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 4);
    EXPECT_EQ(f_size, 4);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_NEAR(output_vec[i], output_ptr[i], 0.1);
    }
}

TEST(deformable_convolution_f32_fw_gpu, basic_deformable_convolution_def_group1) {
    //  Input    : 4x4x4
    //  Trans    : 18x4x4
    //  Output   : 4x4x4
    //  In_offset: 2x2
    //  Filter   : 3x3
    //  Stride   : 1x1
    //  Dilation : 2x2
    //  Group    : 1
    //  Def_group: 1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 4, 4 } });
    auto trans = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 18, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 4, 4, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } });

    set_values(input, { 0.680375f, -0.211234f, 0.566198f, 0.59688f, 0.823295f, -0.604897f, -0.329554f, 0.536459f,
                        -0.444451f, 0.10794f, -0.0452059f, 0.257742f, -0.270431f, 0.0268018f, 0.904459f, 0.83239f,
                        0.271423f, 0.434594f, -0.716795f, 0.213938f, -0.967399f, -0.514226f, -0.725537f, 0.608353f,
                        -0.686642f, -0.198111f, -0.740419f, -0.782382f, 0.997849f, -0.563486f, 0.0258648f, 0.678224f,
                        0.22528f, -0.407937f, 0.275105f, 0.0485743f, -0.012834f, 0.94555f, -0.414966f, 0.542715f,
                        0.0534899f, 0.539828f, -0.199543f, 0.783059f, -0.433371f, -0.295083f, 0.615449f, 0.838053f,
                        -0.860489f, 0.898654f, 0.0519907f, -0.827888f, -0.615572f, 0.326454f, 0.780465f, -0.302214f,
                        -0.871657f, -0.959954f, -0.0845965f, -0.873808f, -0.52344f, 0.941268f, 0.804416f, 0.70184f });

    set_values(trans, { -0.466668f, 0.0795207f, -0.249586f, 0.520497f, 0.0250708f, 0.335448f, 0.0632129f, -0.921439f, -0.124725f,
                        0.86367f, 0.86162f, 0.441905f, -0.431413f, 0.477069f, 0.279958f, -0.291903f, 0.375723f, -0.668052f,
                        -0.119791f, 0.76015f, 0.658402f, -0.339326f, -0.542064f, 0.786745f, -0.29928f, 0.37334f, 0.912936f,
                        0.17728f, 0.314608f, 0.717353f, -0.12088f, 0.84794f, -0.203127f, 0.629534f, 0.368437f, 0.821944f,
                        -0.0350187f, -0.56835f, 0.900505f, 0.840256f, -0.70468f, 0.762124f, 0.282161f, -0.136093f, 0.239193f,
                        -0.437881f, 0.572004f, -0.385084f, -0.105933f, -0.547787f, -0.624934f, -0.447531f, 0.112888f, -0.166997f,
                        -0.660786f, 0.813608f, -0.793658f, -0.747849f, -0.00911188f, 0.52095f, 0.969503f, 0.870008f, 0.36889f,
                        -0.233623f, 0.499542f, -0.262673f, -0.411679f, -0.535477f, 0.168977f, -0.511175f, -0.69522f, 0.464297f,
                        -0.74905f, 0.586941f, -0.671796f, 0.490143f, -0.85094f, 0.900208f, -0.894941f, 0.0431267f, -0.647579f,
                        -0.519875f, 0.595596f, 0.465309f, 0.313127f, 0.93481f, 0.278917f, 0.51947f, -0.813039f, -0.730195f,
                        0.0404202f, -0.843536f, -0.860187f, -0.59069f, -0.077159f, 0.639355f, 0.146637f, 0.511162f, -0.896122f,
                        -0.684386f, 0.999987f, -0.591343f, 0.779911f, -0.749063f, 0.995598f, -0.891885f, 0.74108f, -0.855342f,
                        -0.991677f, 0.846138f, 0.187784f, -0.639255f, -0.673737f, -0.21662f, 0.826053f, 0.63939f, -0.281809f,
                        0.10497f, 0.15886f, -0.0948483f, 0.374775f, -0.80072f, 0.0616159f, 0.514588f, -0.39141f, 0.984457f,
                        0.153942f, 0.755228f, 0.495619f, 0.25782f, -0.929158f, 0.495606f, 0.666477f, 0.850753f, 0.746543f,
                        0.662075f, 0.958868f, 0.487622f, 0.806733f, 0.967191f, 0.333761f, -0.00548297f, -0.672064f, 0.660024f,
                        0.777897f, -0.846011f, 0.299414f, -0.503912f, 0.258959f, -0.541726f, 0.40124f, -0.366266f, -0.342446f,
                        -0.537144f, -0.851678f, 0.266144f, -0.552687f, 0.302264f, 0.021372f, 0.942931f, -0.439916f, 0.0922137f,
                        0.438537f, -0.773439f, -0.0570331f, 0.18508f, 0.888636f, -0.0981649f, -0.327298f, 0.695369f, -0.130973f,
                        -0.993537f, -0.310114f, 0.196963f, 0.666487f, -0.532217f, 0.350952f, -0.0340995f, -0.0361283f, -0.390089f,
                        0.424175f, -0.634888f, 0.243646f, -0.918271f, -0.172033f, 0.391968f, 0.347873f, 0.27528f, -0.305768f,
                        -0.630755f, 0.218212f, 0.254316f, 0.461459f, -0.343251f, 0.480877f, -0.595574f, 0.841829f, 0.369513f,
                        0.306261f, -0.485469f, 0.0648819f, -0.824713f, -0.479006f, 0.754768f, 0.37225f, -0.81252f, -0.777449f,
                        -0.276798f, 0.153381f, 0.186423f, 0.333113f, -0.422444f, 0.551535f, -0.423241f, -0.340716f, -0.620498f,
                        0.968726f, -0.992843f, 0.654782f, -0.337042f, -0.623598f, -0.127006f, 0.917274f, 0.837861f, 0.529743f,
                        0.398151f, -0.757714f, 0.371572f, -0.232336f, 0.548547f, 0.886103f, 0.832546f, 0.723834f, -0.592904f,
                        0.587314f, 0.0960841f, -0.405423f, 0.809865f, 0.819286f, 0.747958f, -0.00371218f, 0.152399f, -0.674487f,
                        -0.452178f, 0.729158f, -0.0152023f, -0.0726757f, 0.697884f, -0.0080452f, -0.417893f, -0.639158f, 0.368357f,
                        0.455101f, -0.721884f, 0.206218f, -0.0151566f, 0.676267f, 0.448504f, -0.643585f, -0.556069f, -0.00294906f,
                        -0.757482f, -0.723523f, -0.279115f, -0.350386f, 0.863791f, 0.816969f, 0.244191f, 0.673656f, 0.636255f,
                        -0.00785118f, -0.330057f, -0.211346f, 0.317662f, 0.217766f, -0.482188f, -0.69754f, -0.85491f, -0.784303f,
                        0.294415f, -0.272803f, -0.423461f, -0.337228f, -0.817703f, -0.145345f, 0.868989f, 0.167141f, -0.469077f });

    set_values(weights, { 0.0f, 0.841471f, 0.909297f, 0.14112f, -0.756802f, -0.958924f, -0.279415f, 0.656987f,
                          0.989358f, 0.412118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.912945f,
                          0.836656f, -0.00885131f, -0.84622f, -0.905578f, -0.132352f, 0.762558f, 0.956376f, 0.270906f,
                          -0.663634f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.745113f, -0.158623f,
                          -0.916522f, -0.831775f, 0.0177019f, 0.850904f, 0.901788f, 0.123573f, -0.768255f, -0.953753f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.304811f, -0.966118f, -0.739181f,
                          0.167356f, 0.920026f, 0.826829f, -0.0265512f, -0.85552f, -0.897928f, -0.114785f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.993889f, -0.629888f, 0.313229f, 0.968364f,
                          0.73319f, -0.176076f, -0.923458f, -0.821818f, 0.0353983f, 0.860069f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.506366f, 0.452026f, 0.994827f, 0.622989f, -0.321622f,
                          -0.970535f, -0.727143f, 0.184782f, 0.926818f, 0.816743f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.580611f, 0.998815f, 0.498713f, -0.459903f, -0.995687f, -0.61604f,
                          0.329991f, 0.97263f, 0.721038f, -0.193473f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.98024f, 0.363171f, -0.587795f, -0.998345f });

    set_values(biases, { -0.491022f, 0.467745f, 0.996469f, 0.609044f });

    std::vector<float> output_vec = {
            0.304297f, -0.385894f, -1.44155f, -0.954556f, -0.260702f, -0.0162079f, 1.05196f, -0.129013f, 0.668587f,
            -1.4058f, -0.0966965f, -0.45043f, -2.23299f, -1.56306f, 0.083207f, -0.42985f, -0.00589353f, 1.08037f,
            1.06648f, 0.0936709f, 1.62321f, 1.50433f, 0.00480294f, -0.0550415f, 0.165425f, 0.146279f, -0.45487f,
            0.370202f, 0.177222f, -1.03958f, -0.744073f, -0.375273f, 0.587801f, 0.120338f, 1.17536f, 1.88443f,
            0.119988f, -0.540461f, -1.7228f, 1.54217f, 0.962263f, -0.0363407f, 0.762274f, 1.32504f, 1.43954f,
            -0.143791f, 1.21981f, 1.71111f, 0.195772f, 0.650412f, 0.474924f, 0.929919f, -0.442715f, 0.462916f,
            -0.210789f, -0.973089f, -0.407542f, 1.11818f, 0.843776f, 0.628229f, 1.29095f, 1.18637f, 0.808982f, 1.43841f };

    topology topology(
            input_layout("input", input.get_layout()),
            input_layout("trans", trans.get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution(
                    "conv",
                    "input",
                    "trans",
                    { "weights" },
                    { "biases" },
                    1,
                    1,
                    { 1, 1, 1, 1 },
                    { 0, 0, -2, -2 },
                    { 1, 1, 2, 2 },
                    { 1, 4, 4, 4 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);
    network.set_input_data("trans", trans);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 4);
    EXPECT_EQ(f_size, 4);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_NEAR(output_vec[i], output_ptr[i], 0.1);
    }
}

TEST(deformable_convolution_f32_fw_gpu, basic_deformable_convolution) {
    //  Input    : 4x4x4
    //  Trans    : 36x4x4
    //  Output   : 4x4x4
    //  In_offset: 2x2
    //  Filter   : 3x3
    //  Stride   : 1x1
    //  Dilation : 2x2
    //  Group    : 1
    //  Def_group: 2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 4, 4 } });
    auto trans = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 36, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 4, 4, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 1, 1 } });

    set_values(input, { 0.680375f, -0.211234f, 0.566198f, 0.59688f, 0.823295f, -0.604897f, -0.329554f, 0.536459f,
                        -0.444451f, 0.10794f, -0.0452059f, 0.257742f, -0.270431f, 0.0268018f, 0.904459f, 0.83239f,
                        0.271423f, 0.434594f, -0.716795f, 0.213938f, -0.967399f, -0.514226f, -0.725537f, 0.608353f,
                        -0.686642f, -0.198111f, -0.740419f, -0.782382f, 0.997849f, -0.563486f, 0.0258648f, 0.678224f,
                        0.22528f, -0.407937f, 0.275105f, 0.0485743f, -0.012834f, 0.94555f, -0.414966f, 0.542715f,
                        0.0534899f, 0.539828f, -0.199543f, 0.783059f, -0.433371f, -0.295083f, 0.615449f, 0.838053f,
                        -0.860489f, 0.898654f, 0.0519907f, -0.827888f, -0.615572f, 0.326454f, 0.780465f, -0.302214f,
                        -0.871657f, -0.959954f, -0.0845965f, -0.873808f, -0.52344f, 0.941268f, 0.804416f, 0.70184f });

    set_values(trans, { -0.466668f, 0.0795207f, -0.249586f, 0.520497f, 0.0250708f, 0.335448f, 0.0632129f, -0.921439f, -0.124725f,
                        0.86367f, 0.86162f, 0.441905f, -0.431413f, 0.477069f, 0.279958f, -0.291903f, 0.375723f, -0.668052f,
                        -0.119791f, 0.76015f, 0.658402f, -0.339326f, -0.542064f, 0.786745f, -0.29928f, 0.37334f, 0.912936f,
                        0.17728f, 0.314608f, 0.717353f, -0.12088f, 0.84794f, -0.203127f, 0.629534f, 0.368437f, 0.821944f,
                        -0.0350187f, -0.56835f, 0.900505f, 0.840256f, -0.70468f, 0.762124f, 0.282161f, -0.136093f, 0.239193f,
                        -0.437881f, 0.572004f, -0.385084f, -0.105933f, -0.547787f, -0.624934f, -0.447531f, 0.112888f, -0.166997f,
                        -0.660786f, 0.813608f, -0.793658f, -0.747849f, -0.00911188f, 0.52095f, 0.969503f, 0.870008f, 0.36889f,
                        -0.233623f, 0.499542f, -0.262673f, -0.411679f, -0.535477f, 0.168977f, -0.511175f, -0.69522f, 0.464297f,
                        -0.74905f, 0.586941f, -0.671796f, 0.490143f, -0.85094f, 0.900208f, -0.894941f, 0.0431267f, -0.647579f,
                        -0.519875f, 0.595596f, 0.465309f, 0.313127f, 0.93481f, 0.278917f, 0.51947f, -0.813039f, -0.730195f,
                        0.0404202f, -0.843536f, -0.860187f, -0.59069f, -0.077159f, 0.639355f, 0.146637f, 0.511162f, -0.896122f,
                        -0.684386f, 0.999987f, -0.591343f, 0.779911f, -0.749063f, 0.995598f, -0.891885f, 0.74108f, -0.855342f,
                        -0.991677f, 0.846138f, 0.187784f, -0.639255f, -0.673737f, -0.21662f, 0.826053f, 0.63939f, -0.281809f,
                        0.10497f, 0.15886f, -0.0948483f, 0.374775f, -0.80072f, 0.0616159f, 0.514588f, -0.39141f, 0.984457f,
                        0.153942f, 0.755228f, 0.495619f, 0.25782f, -0.929158f, 0.495606f, 0.666477f, 0.850753f, 0.746543f,
                        0.662075f, 0.958868f, 0.487622f, 0.806733f, 0.967191f, 0.333761f, -0.00548297f, -0.672064f, 0.660024f,
                        0.777897f, -0.846011f, 0.299414f, -0.503912f, 0.258959f, -0.541726f, 0.40124f, -0.366266f, -0.342446f,
                        -0.537144f, -0.851678f, 0.266144f, -0.552687f, 0.302264f, 0.021372f, 0.942931f, -0.439916f, 0.0922137f,
                        0.438537f, -0.773439f, -0.0570331f, 0.18508f, 0.888636f, -0.0981649f, -0.327298f, 0.695369f, -0.130973f,
                        -0.993537f, -0.310114f, 0.196963f, 0.666487f, -0.532217f, 0.350952f, -0.0340995f, -0.0361283f, -0.390089f,
                        0.424175f, -0.634888f, 0.243646f, -0.918271f, -0.172033f, 0.391968f, 0.347873f, 0.27528f, -0.305768f,
                        -0.630755f, 0.218212f, 0.254316f, 0.461459f, -0.343251f, 0.480877f, -0.595574f, 0.841829f, 0.369513f,
                        0.306261f, -0.485469f, 0.0648819f, -0.824713f, -0.479006f, 0.754768f, 0.37225f, -0.81252f, -0.777449f,
                        -0.276798f, 0.153381f, 0.186423f, 0.333113f, -0.422444f, 0.551535f, -0.423241f, -0.340716f, -0.620498f,
                        0.968726f, -0.992843f, 0.654782f, -0.337042f, -0.623598f, -0.127006f, 0.917274f, 0.837861f, 0.529743f,
                        0.398151f, -0.757714f, 0.371572f, -0.232336f, 0.548547f, 0.886103f, 0.832546f, 0.723834f, -0.592904f,
                        0.587314f, 0.0960841f, -0.405423f, 0.809865f, 0.819286f, 0.747958f, -0.00371218f, 0.152399f, -0.674487f,
                        -0.452178f, 0.729158f, -0.0152023f, -0.0726757f, 0.697884f, -0.0080452f, -0.417893f, -0.639158f, 0.368357f,
                        0.455101f, -0.721884f, 0.206218f, -0.0151566f, 0.676267f, 0.448504f, -0.643585f, -0.556069f, -0.00294906f,
                        -0.757482f, -0.723523f, -0.279115f, -0.350386f, 0.863791f, 0.816969f, 0.244191f, 0.673656f, 0.636255f,
                        -0.00785118f, -0.330057f, -0.211346f, 0.317662f, 0.217766f, -0.482188f, -0.69754f, -0.85491f, -0.784303f,
                        0.294415f, -0.272803f, -0.423461f, -0.337228f, -0.817703f, -0.145345f, 0.868989f, 0.167141f, -0.469077f,
                        0.317493f, 0.523556f, -0.0251462f, -0.685456f, 0.766073f, 0.251331f, 0.0354295f, -0.584313f, 0.115121f,
                        -0.147601f, 0.659878f, -0.211223f, -0.511346f, -0.347973f, 0.45872f, 0.277308f, 0.969689f, -0.323514f,
                        0.79512f, -0.727851f, -0.178424f, -0.989183f, 0.566564f, 0.548772f, -0.412644f, -0.770664f, 0.73107f,
                        0.442012f, -0.901675f, -0.10179f, 0.972934f, 0.415818f, -0.578234f, -0.0522121f, 0.730362f, -0.812161f,
                        -0.800881f, -0.234208f, -0.396474f, 0.31424f, 0.618191f, -0.736596f, -0.896983f, -0.893155f, -0.0845688f,
                        0.561737f, 0.384153f, -0.11488f, -0.761777f, 0.179273f, 0.15727f, 0.0597985f, 0.190091f, -0.276166f,
                        -0.391429f, 0.777447f, -0.0468307f, -0.66036f, 0.219458f, 0.0514942f, 0.237851f, 0.192392f, -0.532688f,
                        0.659617f, -0.85982f, -0.802325f, 0.847456f, -0.660701f, -0.0365333f, -0.549018f, 0.653539f, -0.418343f,
                        -0.285614f, 0.756555f, -0.311498f, 0.629817f, 0.318292f, -0.927345f, -0.485062f, 0.556515f, 0.251928f,
                        0.672207f, -0.383687f, -0.557981f, -0.603959f, 0.224884f, -0.780535f, 0.349211f, 0.564525f, 0.438924f,
                        -0.599295f, -0.197625f, -0.368684f, -0.131983f, -0.538008f, -0.228504f, 0.0656919f, -0.690552f, 0.110795f,
                        -0.970841f, -0.239571f, -0.235666f, -0.389184f, 0.474815f, -0.47911f, 0.299318f, 0.104633f, 0.839182f,
                        0.371973f, 0.619571f, 0.395696f, -0.376099f, 0.291778f, -0.98799f, 0.0659196f, 0.687819f, 0.236894f,
                        0.285385f, 0.0370297f, -0.198582f, -0.275691f, 0.437734f, 0.603793f, 0.355625f, -0.694248f, -0.934215f,
                        -0.872879f, 0.371444f, -0.624767f, 0.237917f, 0.400602f, 0.135662f, -0.997749f, -0.988582f, -0.389522f,
                        -0.476859f, 0.310736f, 0.71511f, -0.637678f, -0.317291f, 0.334681f, 0.758019f, 0.30661f, -0.373541f,
                        0.770028f, -0.62747f, -0.685722f, 0.00692201f, 0.657915f, 0.351308f, 0.80834f, -0.617777f, -0.210957f,
                        0.412133f, 0.737848f, 0.0947942f, 0.477919f, 0.864969f, -0.533762f, 0.853152f, 0.102886f, 0.86684f,
                        -0.0111862f, 0.105137f, 0.878258f, 0.599291f, 0.628277f, 0.188995f, 0.314402f, 0.9906f, 0.871704f,
                        -0.350917f, 0.748619f, 0.178313f, 0.275542f, 0.518647f, 0.550843f, 0.58982f, -0.474431f, 0.208758f,
                        -0.0588716f, -0.666091f, 0.590981f, 0.730171f, 0.746043f, 0.328829f, -0.175035f, 0.223961f, 0.193798f,
                        0.291203f, 0.077113f, -0.703316f, 0.158043f, -0.934073f, 0.401821f, 0.0363014f, 0.665218f, 0.0300982f,
                        -0.774704f, -0.02038f, 0.0206981f, -0.903001f, 0.628703f, -0.230683f, 0.275313f, -0.0957552f, -0.712036f,
                        -0.173844f, -0.505935f, -0.186467f, -0.965087f, 0.435194f, 0.147442f, 0.625894f, 0.165365f, -0.106515f,
                        -0.0452772f, 0.99033f, -0.882554f, -0.851479f, 0.281533f, 0.19456f, -0.554795f, -0.560424f, 0.260486f,
                        0.847025f, 0.475877f, -0.0742955f, -0.122876f, 0.701173f, 0.905324f, 0.897822f, 0.798172f, 0.534027f,
                        -0.332862f, 0.073485f, -0.561728f, -0.0448976f, 0.899641f, -0.0676628f, 0.768636f, 0.934554f, -0.632469f,
                        -0.083922f, 0.560448f, 0.532895f, 0.809563f, -0.484829f, 0.523225f, 0.927009f, -0.336308f, -0.195242f,
                        0.121569f, 0.108896f, 0.244333f, -0.617945f, -0.0440782f, -0.27979f, 0.30776f, 0.833045f, -0.578617f,
                        0.213084f, 0.730867f, -0.780445f, -0.252888f, -0.601995f, 0.29304f, 0.185384f, 0.353108f, 0.192681f,
                        -0.882279f, 0.121744f, 0.127235f, -0.514748f, -0.962178f, -0.312317f, -0.981853f, 0.847385f, 0.202853f,
                        0.541372f, 0.774394f, 0.866545f, -0.653871f, -0.104037f, -0.0245586f, 0.590463f, 0.278018f, 0.931363f });

    set_values(weights, { 0.0f, 0.841471f, 0.909297f, 0.14112f, -0.756802f, -0.958924f, -0.279415f, 0.656987f,
                          0.989358f, 0.412118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.912945f,
                          0.836656f, -0.00885131f, -0.84622f, -0.905578f, -0.132352f, 0.762558f, 0.956376f, 0.270906f,
                          -0.663634f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.745113f, -0.158623f,
                          -0.916522f, -0.831775f, 0.0177019f, 0.850904f, 0.901788f, 0.123573f, -0.768255f, -0.953753f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.304811f, -0.966118f, -0.739181f,
                          0.167356f, 0.920026f, 0.826829f, -0.0265512f, -0.85552f, -0.897928f, -0.114785f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.993889f, -0.629888f, 0.313229f, 0.968364f,
                          0.73319f, -0.176076f, -0.923458f, -0.821818f, 0.0353983f, 0.860069f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.506366f, 0.452026f, 0.994827f, 0.622989f, -0.321622f,
                          -0.970535f, -0.727143f, 0.184782f, 0.926818f, 0.816743f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.580611f, 0.998815f, 0.498713f, -0.459903f, -0.995687f, -0.61604f,
                          0.329991f, 0.97263f, 0.721038f, -0.193473f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.98024f, 0.363171f, -0.587795f, -0.998345f });

    set_values(biases, { -0.491022f, 0.467745f, 0.996469f, 0.609044f });

    std::vector<float> output_vec = {
            -0.119782f, -1.19244f, -0.495079f, -0.760084f, 0.066627f, 0.42253f, 0.365703f, 0.62729f, 0.380708f,
            0.0900432f, -0.866731f, -0.784469f, -2.18692f, -1.73267f, 0.251761f, -0.791547f, 0.634862f, 0.646589f,
            -0.321454f, 0.575675f, 0.98983f, -0.445829f, 0.523965f, -0.346374f, 0.127803f, -2.13572f, -0.796409f,
            1.5734f, -0.972705f, -1.88344f, -1.04588f, -0.0209212f, 0.78641f, -0.3878f, 0.151791f, 2.08673f, 0.698802f,
            0.584678f, -0.78199f, -0.352576f, 1.03862f, 0.229792f, -0.223219f, 1.02365f, 1.45293f, 0.0561579f, 1.95182f,
            1.59586f, 0.773778f, 0.648415f, 1.65464f, 1.311f, 0.326254f, -0.447391f, -0.858153f, -0.702836f, -0.589441f,
            1.18929f, 0.382556f, 0.499048f, 1.16212f, 1.62688f, 1.31246f, 1.82684f };

    topology topology(
            input_layout("input", input.get_layout()),
            input_layout("trans", trans.get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution(
                    "conv",
                    "input",
                    "trans",
                    { "weights" },
                    { "biases" },
                    1,
                    2,
                    { 1, 1, 1, 1 },
                    { 0, 0, -2, -2 },
                    { 1, 1, 2, 2 },
                    { 1, 4, 4, 4 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);
    network.set_input_data("trans", trans);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 4);
    EXPECT_EQ(x_size, 4);
    EXPECT_EQ(f_size, 4);
    EXPECT_EQ(b_size, 1);

    for (size_t i = 0; i < output_vec.size(); ++i) {
        EXPECT_NEAR(output_vec[i], output_ptr[i], 0.1);
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_no_bias) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32,format::yxfb,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 3, 2 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    VVF<float> output_vec = {
        { 20.0f, 27.0f, 38.0f },
        { 17.0f, 19.0f, 19.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }, { 1,1,1,2 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_convolution_int8_no_bias) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32,format::bfyx,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::i8,format::bfyx,{ 1, 1, 3, 2 } });

    set_values(input, { 1.1f, 2.4f, 3.5f, 4.5f, 5.8f,
                        2.9f, 2.3f, 3.5f, 4.4f, 6.6f,
                        3.8f, 3.9f, 3.4f, 5.1f, 1.4f,
                        1.8f, 1.1f, 1.2f, 1.2f, 1.9f });
    set_values<int8_t>(weights, { 1, 2, 1,
                                  2, 1, 2 });
    VVF<float> output_vec = {
        { 25.0f, 31.0f, 46.0f },
        { 22.0f, 20.0f, 21.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("to_int","input", { data_types::i8,format::bfyx,{ 1, 1, 5, 4 } }),
        data("weights", weights),
        convolution("conv", "to_int", { "weights" }, { 1,1,1,2 }),
        reorder("output", "conv", { data_types::f32,format::bfyx,{ 1, 1, 3, 2 } }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D_no_bias) {
    //  data is similar as in basic_convolution_no_bias

    //  Filter : 2x3x1
    //  Stride : 2x1x1
    //  Input  : 4x5x1
    //  Output : 2x3x1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 2 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });

    VVF<float> output_vec = {
        { 20.0f, 27.0f, 38.0f },
        { 17.0f, 19.0f, 19.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }, { 1,1,1,2 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int z_size = output_layout.size.spatial[2];
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(z_size, 1);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D) {
    //  Input  : 4x4x4
    //  Filter : 2x2x2
    //  Output : 3x3x3
    //
    //  Input:
    //  1  0  1  0
    //  1  1  3  1
    //  1  1  0  2
    //  0  2  1  1
    //
    //  1  0  0  1
    //  2  0  1  2
    //  3  1  1  1
    //  0  0  3  1
    //
    //  2  0  1  1
    //  3  3  1  0
    //  2  1  1  0
    //  3  2  1  2
    //
    //  1  0  2  0
    //  1  0  3  3
    //  3  1  0  0
    //  1  1  0  2
    //
    //  Filter:
    //  0  1
    //  0  0
    //
    //  2  1
    //  0  0

    //  Output:
    //  2  1  1
    //  5  4  5
    //  8  3  5
    //
    //  4  1  4
    //  9  8  4
    //  6  4  3
    //
    //  2  3  5
    //  5  4  9
    //  8  3  0
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 4, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
    });

    set_values(weights, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(biases, { 1.0f });

    VVVF<float> output_vec = {
        {
            { 3.0f,   2.0f,   2.0f },
            { 6.0f,   5.0f,   6.0f },
            { 9.0f,   4.0f,   6.0f }
        },
        {
            { 5.0f,   2.0f,   5.0f },
            { 10.0f,   9.0f,   5.0f },
            { 7.0f,   5.0f,   4.0f }
        },
        {
            { 3.0f,   4.0f,   6.0f },
            { 6.0f,   5.0f,   10.0f },
            { 9.0f,   4.0f,   1.0f }
        }
    };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int z_size = output_layout.size.spatial[2];
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(z_size, 3);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 3);

    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int z = 0; z < z_size; ++z) {
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_EQ(output_vec[z][y][x], output_ptr[z * y_size * x_size + y * x_size + x]);
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D_split2) {
    //  data is similar as in basic_convolution3D
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 2, 4, 4, 4 } });
    auto weights_1 = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });
    auto weights_2 = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 1, 2, 2, 2 } });
    auto biases_1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1, 1 } });
    auto biases_2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
    });

    set_values(weights_1, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(weights_2, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(biases_1, { 1.0f });
    set_values(biases_2, { 2.0f });

    VVVVF<float> output_vec = {
        {
            {
                { 3.0f,   2.0f,   2.0f },
                { 6.0f,   5.0f,   6.0f },
                { 9.0f,   4.0f,   6.0f }
            },
            {
                { 5.0f,   2.0f,   5.0f },
                { 10.0f,   9.0f,   5.0f },
                { 7.0f,   5.0f,   4.0f }
            },
            {
                { 3.0f,   4.0f,   6.0f },
                { 6.0f,   5.0f,   10.0f},
                { 9.0f,   4.0f,   1.0f }
            },
        },
        {
            {
                { 4.0f,   3.0f,   3.0f },
                { 7.0f,   6.0f,   7.0f },
                { 10.0f,  5.0f,   7.0f }
            },
            {
                { 6.0f,   3.0f,   6.0f },
                { 11.0f,  10.0f,  6.0f },
                { 8.0f,   6.0f,   5.0f }
            },
            {
                { 4.0f,   5.0f,   7.0f },
                { 7.0f,   6.0f,  11.0f },
                { 10.0f,  5.0f,   2.0f }
            },
        }
    };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights_1", weights_1),
        data("biases_1", biases_1),
        data("weights_2", weights_2),
        data("biases_2", biases_2),
        convolution("conv", "input", { "weights_1",  "weights_2" }, { "biases_1",  "biases_2" }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int z_size = output_layout.size.spatial[2];
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(z_size, 3);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(b_size, 1);
    EXPECT_EQ(f_size, 2);
    for (int f = 0; f < f_size; ++f) {
        for (int z = 0; z < z_size; ++z) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    EXPECT_EQ(output_vec[f][z][y][x],
                        output_ptr[f * z_size * y_size * x_size + z * y_size * x_size + y * x_size + x]);
                }
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution3D_group2) {
    //  data is similar as in basic_convolution3D_split2
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 1, 2, 4, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfzyx,{ 2, 1, 2, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1, 1 } });

    set_values(input, {
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        1.0f,  0.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  3.0f,  1.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
        0.0f,  2.0f,  1.0f,  1.0f,
        1.0f,  0.0f,  0.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  2.0f,
        3.0f,  1.0f,  1.0f,  1.0f,
        0.0f,  0.0f,  3.0f,  1.0f,
        2.0f,  0.0f,  1.0f,  1.0f,
        3.0f,  3.0f,  1.0f,  0.0f,
        2.0f,  1.0f,  1.0f,  0.0f,
        3.0f,  2.0f,  1.0f,  2.0f,
        1.0f,  0.0f,  2.0f,  0.0f,
        1.0f,  0.0f,  3.0f,  3.0f,
        3.0f,  1.0f,  0.0f,  0.0f,
        1.0f,  1.0f,  0.0f,  2.0f,
    });

    set_values(weights, {
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
        0.0f,  1.0f,
        0.0f,  0.0f,
        2.0f,  1.0f,
        0.0f,  0.0f,
    });

    set_values(biases, { 1.0f, 2.0f });

    VVVVF<float> output_vec = {
        {
            {
                { 3.0f,   2.0f,   2.0f },
                { 6.0f,   5.0f,   6.0f },
                { 9.0f,   4.0f,   6.0f }
            },
            {
                { 5.0f,   2.0f,   5.0f },
                { 10.0f,   9.0f,   5.0f },
                { 7.0f,   5.0f,   4.0f }
            },
            {
                { 3.0f,   4.0f,   6.0f },
                { 6.0f,   5.0f,   10.0f },
                { 9.0f,   4.0f,   1.0f }
            },
        },
        {
            {
                { 4.0f,   3.0f,   3.0f },
                { 7.0f,   6.0f,   7.0f },
                { 10.0f,  5.0f,   7.0f }
            },
            {
                { 6.0f,   3.0f,   6.0f },
                { 11.0f,  10.0f,  6.0f },
                { 8.0f,   6.0f,   5.0f }
            },
            {
                { 4.0f,   5.0f,   7.0f },
                { 7.0f,   6.0f,  11.0f },
                { 10.0f,  5.0f,   2.0f }
            },
        }
    };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int z_size = output_layout.size.spatial[2];
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfzyx);
    EXPECT_EQ(b_size, 1);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(z_size, 3);
    EXPECT_EQ(y_size, 3);
    EXPECT_EQ(x_size, 3);
    for (int f = 0; f < f_size; ++f) {
        for (int z = 0; z < z_size; ++z) {
            for (int y = 0; y < y_size; ++y) {
                for (int x = 0; x < x_size; ++x) {
                    EXPECT_EQ(output_vec[f][z][y][x],
                        output_ptr[f * z_size * y_size * x_size + z * y_size * x_size + y * x_size + x]);
                }
            }
        }
    }
}

TEST(convolution_f32_fw_gpu, with_output_size_same_input) {

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 4, 320, 320 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 64, 4, 7, 7 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 64, 4, 7, 7 } });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("weights2", weights2),
        convolution::create_with_output_size("conv1", "input", { "weights" }, {1, 64, 160, 160}, {1, 1, 2, 2}, {0, 0, -3, -3}),
        convolution::create_with_output_size("conv2", "input", { "weights2" }, {1, 64, 320, 320}, {1, 1, 1, 1}, {0, 0, -3, -3})
        );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(2));
    EXPECT_EQ(outputs.begin()->first, "conv1");
    EXPECT_EQ(outputs.rbegin()->first, "conv2");
}

TEST(convolution_f32_fw_gpu, three_convolutions_same_weights) {
    //  Filter : 1x1
    //  Input  : 2x2
    //  Output : 2x2
    //
    //  Input:
    //  1  1   1  1
    //  1  1   1  1
    //
    //  Filter:
    //  1
    //
    //  Output:
    //  8  8   8  8
    //  8  8   8  8

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, {1,2,2,2} });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 2,2,1,1 } });

    set_values(input, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution("conv1", "input", { "weights" }),
        convolution("conv2", "conv1", { "weights" }),
        convolution("conv3", "conv2", { "weights" })
    );

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output_memory = outputs.at("conv3").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];

    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 2);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_FLOAT_EQ(8.0f, output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  Output:
    // 21  28  39
    // 18  20  20
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(weights, { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 21.0f, 28.0f, 39.0f },
        { 18.0f, 20.0f, 20.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution( "conv", "input", { "weights" }, { "biases" }, { 0,0,1,2 }));

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_bfyx_weights_as_input_layout) {
    //Same params as convolution_f32_fw_gpu, basic_convolution but with bfyx optimized data and weights set as input_layout
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,
    { 1, 1, 5, 4 }
    });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,
    { 1, 1, 3, 2 }
    });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,
    { 1, 1, 1, 1 }
    });
    set_values(input,
    { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f, 3.0f, 3.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }
    );
    set_values(weights,
    { 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f }
    );
    set_values(biases,
    { 1.0f }
    );
    VVF<float> output_vec = {
        { 21.0f, 28.0f, 39.0f }
        ,
        { 18.0f, 20.0f, 20.0f }
    };
    topology topology(
        input_layout("input", input.get_layout()),
        input_layout("weights", weights.get_layout()),
        input_layout("biases", biases.get_layout()),
        convolution("conv", "input",
        { "weights" }
            ,
            { "biases" }
            ,
            { 0,0,1,2 }
    ));
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);
    network.set_input_data("input", input);
    network.set_input_data("weights", weights);
    network.set_input_data("biases", biases);
    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_input_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : 2x1
    //  Output : 6x5
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1
    //  2  4  6  8  5
    //  4  8 11 15  9
    //  6 11 12 16 10
    //  4  7  7  9  6
    //  1  1  1  1  1
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 2.0f, 4.0f, 6.0f, 8.0f, 5.0f },
        { 4.0f, 8.0f, 11.0f, 15.0f, 9.0f },
        { 6.0f, 11.0f, 12.0f, 16.0f, 10.0f },
        { 4.0f, 7.0f, 7.0f, 9.0f, 6.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,-1,-2 },
            { 1, 1, 1, 1 },
            padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 6);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_convolution_sym_input_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 2x1
    //  Output : 6x5
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1
    //  2  4  6  8  5
    //  4  8 11 15  9
    //  6 11 12 16 10
    //  4  7  7  9  6
    //  1  1  1  1  1
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 2.0f, 4.0f, 6.0f, 8.0f, 5.0f },
        { 4.0f, 8.0f, 11.0f, 15.0f, 9.0f },
        { 6.0f, 11.0f, 12.0f, 16.0f, 10.0f },
        { 4.0f, 7.0f, 7.0f, 9.0f, 6.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,0,0 },
            { 1, 1, 1, 1 },
            { 0,0,1,2 },
            { 0,0,1,2 },
            padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 6);
    EXPECT_EQ(x_size, 5);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_asym_input_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 3x2
    //  Output : 7x6
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  1  2  3  4  z  z
    //  z  2  2  3  4  z  z
    //  z  3  3  3  5  z  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1
    //  2  4  6  8  5  1
    //  4  8 11 15  9  1
    //  6 11 12 16 10  1
    //  4  7  7  9  6  1
    //  1  1  1  1  1  1
    //  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f },
        { 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f },
        { 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f },
        { 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,0,0 },
            { 1, 1, 1, 1 },
            { 0,0,1,2 },
            { 0,0,2,3 },
            padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 7);
    EXPECT_EQ(x_size, 6);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_sym_input_padding_with_input_offset) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 2x1
    //  Input offset: 2x1
    //  Output : 10x7
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  1  2  3  4  z  z
    //  z  z  2  2  3  4  z  z
    //  z  z  3  3  3  5  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //  1  2  4  6  8  5  1
    //  1  4  8 11 15  9  1
    //  1  6 11 12 16 10  1
    //  1  4  7  7  9  6  1
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f },
        { 1.0f, 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f },
        { 1.0f, 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f },
        { 1.0f, 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,-1,-2 },
            { 1, 1, 1, 1 },
            { 0,0,1,2 },
            { 0,0,1,2 },
            padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 10);
    EXPECT_EQ(x_size, 7);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_asym_input_padding_with_input_offset) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : above 2x1, below 3x2
    //  Input offset: 2x1
    //  Output : 11x8
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  1  2  3  4  z  z  z
    //  z  z  2  2  3  4  z  z  z
    //  z  z  3  3  3  5  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //  z  z  z  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  2  4  6  8  5  1  1
    //  1  4  8 11 15  9  1  1
    //  1  6 11 12 16 10  1  1
    //  1  4  7  7  9  6  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f, 1.0f },
        { 1.0f, 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f, 1.0f },
        { 1.0f, 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f, 1.0f },
        { 1.0f, 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,-1,-2 },
            { 1, 1, 1, 1 },
            { 0,0,1,2 },
            { 0,0,2,3 },
            padding{ { 0,0,0,0 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 11);
    EXPECT_EQ(x_size, 8);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }
}

TEST(convolution_f32_fw_gpu, basic_convolution_input_and_output_padding) {
    //  Filter : 2x2
    //  Stride : 1x1
    //  Input  : 3x4
    //  Input padding : 2x1
    //  Output : 8x9
    //  Padding: Zero
    //
    //  Input:
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //  z  1  2  3  4  z
    //  z  2  2  3  4  z
    //  z  3  3  3  5  z
    //  z  z  z  z  z  z
    //  z  z  z  z  z  z
    //
    //  Filter:
    //  1  1
    //  1  1
    //
    //  Output:
    //  1  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1  1
    //  1  1  2  4  6  8  5  1  1
    //  1  1  4  8 11 15  9  1  1
    //  1  1  6 11 12 16 10  1  1
    //  1  1  4  7  7  9  6  1  1
    //  1  1  1  1  1  1  1  1  1
    //  1  1  1  1  1  1  1  1  1
    //
    //  Bias:
    //  1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f, 4.0f, 3.0f, 3.0f, 3.0f, 5.0f });
    set_values(weights, { 1.0f, 1.0f, 1.0f, 1.0f });
    set_values(biases, { 1.0f });
    VVF<float> output_vec = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 5.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 4.0f, 8.0f, 11.0f, 15.0f, 9.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 6.0f, 11.0f, 12.0f, 16.0f, 10.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 4.0f, 7.0f, 7.0f, 9.0f, 6.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } };

    const int x_pad = 2;
    const int y_pad = 1;
    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,1,1 },
            { 0,0,-1,-2 },
            { 1, 1, 1, 1 },
            padding{ { 0,0,-x_pad,-y_pad }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_memory = outputs.at("conv").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_size = output_layout.get_buffer_size();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_size.spatial[1];
    int x_size = output_size.spatial[0];
    int f_size = output_size.feature[0];
    int b_size = output_size.batch[0];
    EXPECT_EQ(output_layout.format, format::yxfb);
    EXPECT_EQ(y_size, 8);
    EXPECT_EQ(x_size, 9);
    EXPECT_EQ(f_size, 1);
    EXPECT_EQ(b_size, 1);

    for (int y = y_pad; y < y_size - y_pad; ++y)
    {
        for (int x = x_pad; x < x_size - x_pad; ++x)
        {
            EXPECT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    //VVF temp_vec(y_size, VF(x_size, 0.0f));
    //for (int y = 0; y < y_size; ++y) {
    //    for (int x = 0; x < x_size; ++x) {
    //        temp_vec[y][x] = output_ptr[y * x_size + x];
    //    }
    //}
    //print_2d(temp_vec);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad_random) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
    //  Input:
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //  rnd  rnd  rnd  rnd
    //
    //  Filter
    //  rnd  rnd
    //  rnd  rnd
    //
    //  Bias
    //  rnd
    //
    //  Output:
    //  rnd  rnd
    //  rnd  rnd

    size_t batch = 1, input_f = 1, input_y = 4, input_x = 4;

    VVVVF<float> input_rnd = generate_random_4d<float>(batch, input_f, input_y, input_x, -10, 10);
    VF<float> input_rnd_vec = flatten_4d<float>(format::yxfb, input_rnd);
    VVVVF<float> filter_rnd = generate_random_4d<float>(1, 1, 2, 2, -10, 10);
    VF<float> filter_rnd_vec = flatten_4d<float>(format::bfyx, filter_rnd);
    VF<float> bias_rnd = generate_random_1d<float>(1, -10, 10);
    VVVVF<float> output_rnd(batch, VVVF<float>(filter_rnd.size()));
    for (size_t b = 0; b < output_rnd.size(); ++b) {
        for (size_t of = 0; of < filter_rnd.size(); ++of) {
            output_rnd[b][of] = reference_convolve<float>(input_rnd[b], filter_rnd[of], 2, 2, bias_rnd[of]);
        }
    }
    VF<float> output_rnd_vec = flatten_4d<float>(format::yxfb, output_rnd);

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32,  format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32,  format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32,  format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", {"weights"}, {"biases"}, {1,1,2,2})
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    for (size_t i = 0; i < output_rnd.size(); ++i) {
        float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad_random) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  rnd  rnd    rnd  rnd
    //  rnd  rnd    rnd  rnd
    //
    //  Filter:
    //  rnd  rnd
    //  rnd  rnd
    //
    //  Bias:
    //  rnd
    //
    //  Output:
    //  rnd  rnd

    size_t batch = 2, input_f = 1, input_y = 2, input_x = 2;

    VVVVF<float> input_rnd = generate_random_4d<float>(batch, input_f, input_y, input_x, -10, 10);
    VF<float> input_rnd_vec = flatten_4d<float>(format::yxfb, input_rnd);
    VVVVF<float> filter_rnd = generate_random_4d<float>(1, 1, 2, 2, -10, 10);
    VF<float> filter_rnd_vec = flatten_4d<float>(format::bfyx, filter_rnd);
    VF<float> bias_rnd = generate_random_1d<float>(1, -10, 10);
    VVVVF<float> output_rnd(batch, VVVF<float>(filter_rnd.size()));
    for (size_t b = 0; b < output_rnd.size(); ++b) {
        for (size_t of = 0; of < filter_rnd.size(); ++of) {
            output_rnd[b][of] = reference_convolve<float>(input_rnd[b], filter_rnd[of], 2, 2, bias_rnd[of]);
        }
    }
    VF<float> output_rnd_vec = flatten_4d<float>(format::yxfb, output_rnd);

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, input_rnd_vec);
    set_values(weights, filter_rnd_vec);
    set_values(biases, bias_rnd);

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    for (size_t i = 0; i < output_rnd.size(); ++i) {
        float x = float_round(output_rnd_vec[i]), y = float_round(output_ptr[i]);
        EXPECT_FLOAT_EQ(x, y) << "random seed = " << random_seed << std::endl;
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x1x1_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //
    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  8  0.5
    //  6  9

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.5f, output_ptr[1]);
    EXPECT_FLOAT_EQ(6.0f, output_ptr[2]);
    EXPECT_FLOAT_EQ(9.0f, output_ptr[3]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in2x2x1x2_nopad) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x2x1x2
    //  Output : 1x1x1x2
    //
    //  Input:
    //  0.5   1.5    2.3 -0.4
    //  2.0  -4.0    1.0  3.0
    //
    //  Filter:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias:
    //  -1
    //
    //  Output:
    //  3.65 -5.36
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 1, 1 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 0.5f, 2.3f, 1.5f, -0.4f, 2.0f, 1.0f, -4.0f, 3.0f });
    set_values(weights, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases, { -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 } )
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(3.65f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.36f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz2x1x2x1_in1x2x1_nopad) {
    //  Filter : 1x2x1x2x1
    //  Input  : 1x1x2x1
    //  Output : 1x2x1x1
    //
    //  Input:
    //  1.0    2.0
    //
    // Filter:
    //   1.0    2.0  ofm=0
    //  -1.0   -2.0  ofm=1
    //
    //  Bias:
    //  0.1 -0.2
    //
    //  Output:
    //   5.1  f=0
    //  -5.2  f=1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 1, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, { 1.0f, 2.0f });
    set_values(weights, { 1.0f, 2.0f, -1.0f, -2.0f });
    set_values(biases, { 0.1f, -0.2f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,5,5 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(5.1f, output_ptr[0]);
    EXPECT_FLOAT_EQ(-5.2f, output_ptr[1]);
}

TEST(convolution_f32_fw_gpu, basic_ofm_wsiz3x2x2x1_in2x2x1_nopad) {
    //  Filter : 1x3x2x2x1
    //  Input  : 1x2x2x1
    //  Output : 1x3x1x1
    //
    //  Input:
    //  1.0    2.0  f=0
    //  3.0    4.0  f=1
    //
    // Filter:
    //   1.0    2.0  ifm=0  ofm=0
    //   3.0    4.0  ifm=1
    //
    //   5.0    6.0  ifm=0  ofm=1
    //   7.0    8.0  ifm=1
    //
    //   9.0   10.0  ifm=0  ofm=2
    //  11.0   12.0  ifm=1
    //  Bias:
    //   -5     -6     -7
    //
    //  Output:
    //   25.0  f=0
    //   64,0  f=1
    //  103.0  f=2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 1, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 3, 1, 1 } });

    set_values(input, { 1.0f, 3.0f, 2.0f, 4.0f });
    set_values(weights, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f });
    set_values(biases, { -5.0f, -6.0f, -7.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,5,5 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(25.0f, output_ptr[0]);
    EXPECT_FLOAT_EQ(64.0f, output_ptr[1]);
    EXPECT_FLOAT_EQ(103.0f, output_ptr[2]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2x1x3_wstr2x2_in2x2x1x1_nopad) {
    //  Filter : 2x2x1x3
    //  Stride : 2x2
    //  Input  : 2x2x1x1
    //  Output : 1x1x3x1
    //
    //  Input:
    //  -2.3 -0.1
    //   3.1  1.9
    //
    //  Filter:
    //  -1.1  1.5       0.1  0.2        2.0  -1.0
    //   0.5 -0.5       0.4  0.7        2.5  -1.5
    //
    //  Bias:
    //  0.1 -0.2 0.3
    //
    //  Output:
    //   0.7
    //   2.12
    //   3.08

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 1, 1 }, 3 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 3, 1, 1 } });

    set_values(input, { -2.3f, -0.1f, 3.1f, 1.9f });
    set_values(weights, { -1.1f, 1.5f, 0.5f, -0.5f, 0.1f, 0.2f, 0.4f, 0.7f, 2.0f, -1.0f, 2.5f, -1.5f });
    set_values(biases, { 0.1f, -0.2f, 0.3f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_TRUE(are_equal(3.08f, output_ptr[0]));
    EXPECT_TRUE(are_equal(2.12f, output_ptr[1]));
    EXPECT_TRUE(are_equal(0.7f,  output_ptr[2]));
}

TEST(convolution_f32_fw_gpu, wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //  Filter  : 3x3
    //  Stride  : 2x2
    //  Input   : 2x2
    //  Output  : 1x1
    //  Padding : zero
    //
    //  Input:
    //  -0.5   1.0   padd
    //   0.5   2.0   padd
    //  padd  padd   padd
    //
    //  Filter
    //  -2    0.5  3.5
    //   1.5  4   -5
    //   0.5  1.5 -1.5
    //
    //  Bias
    //  2
    //
    //  Output:
    //  12.25
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 1,1,2,2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(12.25f, output_ptr[0]);
}

TEST(convolution_f32_fw_gpu, offsets_wsiz3x3_wstr2x2_in2x2x1x1_zeropad) {
    //   Filter       : 3x3
    //   Stride       : 2x2
    //   Input        : 2x2
    //   Input offset : -1x-1
    //   Output       : 2x2
    //   Output offset: 1x1
    //   Padding      : zero
    //
    //   Input:
    //   padd padd  padd
    //   padd -0.5   1
    //   padd  0.5   2.0
    //
    //   Filter
    //   -2    0.5  3.5
    //    1.5  4   -5
    //    0.5  1.5 -1.5
    //
    //   Bias
    //   2
    //
    //   Output:
    //   rnd   rnd
    //   rnd   2.0
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 2, 2 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 3, 3 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { -0.5f, 1.0f, 0.5f, 2.0f });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f, 4.0f, -5.0f, 0.5f, 1.5f, -1.5f });
    set_values(biases, { 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,2,2 },
            { 0,0,-1,-1 },
            { 1, 1, 1, 1 },
            padding{ { 0,0,1,1 }, 0 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-7.25f, output_ptr[4]);
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2) {
    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4x2
    //  Output : 2x2x2
    //
    //  Input:
    //  f0: -0.5   1     0.5  2
    //       1.5  -0.5   0   -1
    //       0.5   0.5  -1    1
    //       0.5   2     1.5 -0.5
    //
    //  f1:  0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //       0.5   1.5   2.3 -0.4
    //       2.0  -4.0   1.0  3.0
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  3.65 0.5 -5.36
    //   6  3.65 9   -5.36

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 0,0,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2) {
    //  2x Filter : 2x2
    //  Stride : 2x2
    //  Input  : 2x4x4x2
    //  Output : 2x2x2x2
    //
    //  Input:
    //  f0b0: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f0b1: -0.5   1     0.5  2
    //         1.5  -0.5   0   -1
    //         0.5   0.5  -1    1
    //         0.5   2     1.5 -0.5
    //
    //  f1b0:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //  f1b1:  0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //         0.5   1.5   2.3 -0.4
    //         2.0  -4.0   1.0  3.0
    //
    //
    //  Filter1:
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias1:
    //  2
    //
    //  Filter2:
    //  -1.2  1.5
    //   0.5 -0.5
    //
    //  Bias2:
    //  -1

    //  Output:
    //   8  8 3.65 3.65 0.5  0.5 -5.36 -5.36
    //   6  6 3.65 3.65 9    9   -5.36 -5.36

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 2, 2, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 2,{ 2, 2 }, 2 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
       -0.5f, -0.5f,  0.5f,  0.5f,  1.0f,  1.0f,  1.5f,  1.5f,  0.5f,  0.5f,  2.3f,  2.3f,  2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, -0.5f, -0.5f, -4.0f, -4.0f,  0.0f,  0.0f,  1.0f,  1.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  1.5f,  1.5f, -1.0f, -1.0f,  2.3f,  2.3f,  1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f,  2.0f,  2.0f, -4.0f, -4.0f,  1.5f,  1.5f,  1.0f,  1.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });
    set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases1, { 2.0f });
    set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
    set_values(biases2, { -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f,   get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(0.5f,   get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 8));
    EXPECT_FLOAT_EQ(6.0f,   get_value<float>(output_ptr, 9));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 10));
    EXPECT_FLOAT_EQ(3.65f,  get_value<float>(output_ptr, 11));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 12));
    EXPECT_FLOAT_EQ(9.0f,   get_value<float>(output_ptr, 13));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 14));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 15));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_group2) {
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights, {
        -2.0f, 0.5f, 3.5f, 1.5f,
        -1.2f, 1.5f, 0.5f, -0.5f
    });
    set_values(biases, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            2, // number of groups
            { 0,0,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(0.5f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(9.0f, get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_group2_bfyx) {
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x1_nopad_split2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1, 2, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, {
        -0.5f,  0.5f,  1.0f,  1.5f,  0.5f,  2.3f,  2.0f, -0.4f,
        1.5f,  2.0f, -0.5f, -4.0f,  0.0f,  1.0f, -1.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  1.5f, -1.0f,  2.3f,  1.0f, -0.4f,
        0.5f,  2.0f,  2.0f, -4.0f,  1.5f,  1.0f, -0.5f,  3.0f
    });
    set_values(weights, {
        -2.0f, 0.5f, 3.5f, 1.5f,
        -1.2f, 1.5f, 0.5f, -0.5f
    });
    set_values(biases, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        reorder("input_1", "input", { data_types::f32,format::bfyx,{ 1, 2, 4, 4 } }),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input_1",
            { "weights" },
            { "biases" },
            2, // number of groups
            { 0,0,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(0.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(9.0f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_group2) {
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 2, 4, 4 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values(input, {
        -0.5f, -0.5f,  0.5f,  0.5f,  1.0f,  1.0f,  1.5f,  1.5f,  0.5f,  0.5f,  2.3f,  2.3f,  2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, -0.5f, -0.5f, -4.0f, -4.0f,  0.0f,  0.0f,  1.0f,  1.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  1.5f,  1.5f, -1.0f, -1.0f,  2.3f,  2.3f,  1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f,  2.0f,  2.0f, -4.0f, -4.0f,  1.5f,  1.5f,  1.0f,  1.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });
    set_values(weights, {
        -2.0f, 0.5f, 3.5f, 1.5f,
        -1.2f, 1.5f, 0.5f, -0.5f
    });
    set_values(biases, { 2.0f, -1.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            2, // number of groups
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(0.5f, get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(0.5f, get_value<float>(output_ptr, 5));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 6));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 7));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 8));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 9));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 10));
    EXPECT_FLOAT_EQ(3.65f, get_value<float>(output_ptr, 11));
    EXPECT_FLOAT_EQ(9.0f, get_value<float>(output_ptr, 12));
    EXPECT_FLOAT_EQ(9.0f, get_value<float>(output_ptr, 13));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 14));
    EXPECT_FLOAT_EQ(-5.36f, get_value<float>(output_ptr, 15));
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2 but with batch 1

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f,
        1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f,
        0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f,
        2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f,
        -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f,
        0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
        0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f,
        -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f,
        1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f,
        2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f,
        1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f,
        -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    for (uint32_t i = 0; i < 8; i++)
    {
        auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

        set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
        set_values(biases1, { 2.0f });
        set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
        set_values(biases2, { -1.0f });

        primitive_id weights_id = "weights_" + std::to_string(i);
        primitive_id weights2_id = "weights2_" + std::to_string(i);
        primitive_id bias_id = "biases_" + std::to_string(i);
        primitive_id bias2_id = "biases2_" + std::to_string(i);

        weights_vec.push_back(weights_id);
        weights_vec.push_back(weights2_id);
        bias_vec.push_back(bias_id);
        bias_vec.push_back(bias2_id);

        topology.add(
            data(weights_id, weights1),
            data(bias_id, biases1),
            data(weights2_id, weights2),
            data(bias2_id, biases2)
        );

    }

    topology.add(
        convolution(
            "conv",
            "input",
            weights_vec,
            bias_vec,
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f,
        0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f,
        6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f,
        9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt_bfyx) {
    //  Test for depthwise separable optimization, there are 16 weights and biases (split 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2 but with batch 1
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
    });

    topology topology(input_layout("input", input.get_layout()));

    std::vector<primitive_id> weights_vec;
    std::vector<primitive_id> bias_vec;

    for (uint32_t i = 0; i < 8; i++)
    {
        auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
        auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

        set_values(weights1, { -2.0f, 0.5f, 3.5f, 1.5f });
        set_values(biases1, { 2.0f });
        set_values(weights2, { -1.2f, 1.5f, 0.5f, -0.5f });
        set_values(biases2, { -1.0f });

        primitive_id weights_id = "weights_" + std::to_string(i);
        primitive_id weights2_id = "weights2_" + std::to_string(i);
        primitive_id bias_id = "biases_" + std::to_string(i);
        primitive_id bias2_id = "biases2_" + std::to_string(i);

        weights_vec.push_back(weights_id);
        weights_vec.push_back(weights2_id);
        bias_vec.push_back(bias_id);
        bias_vec.push_back(bias2_id);

        topology.add(
            data(weights_id, weights1),
            data(bias_id, biases1),
            data(weights2_id, weights2),
            data(bias2_id, biases2)
        );

    }

    topology.add(
        convolution(
            "conv",
            "input",
            weights_vec,
            bias_vec,
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_group16) {
    //  Test for grouped convolution, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt

    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f,
        1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f, 1.0f,  1.0f,  1.5f,  1.5f,
        0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f, 0.5f,  0.5f,  2.3f,  2.3f,
        2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f, 2.0f,  2.0f, -0.4f, -0.4f,
        1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f, 1.5f,  1.5f,  2.0f,  2.0f,
        -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f, -0.5f, -0.5f, -4.0f, -4.0f,
        0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f, 0.0f,  0.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f, -1.0f, -1.0f,  3.0f,  3.0f,
        0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
        0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f, 0.5f,  0.5f,  1.5f,  1.5f,
        -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f, -1.0f, -1.0f,  2.3f,  2.3f,
        1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f, 1.0f,  1.0f, -0.4f, -0.4f,
        0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f, 0.5f,  0.5f,  2.0f,  2.0f,
        2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f, 2.0f,  2.0f, -4.0f, -4.0f,
        1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f, 1.5f,  1.5f,  1.0f,  1.0f,
        -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f, -0.5f, -0.5f,  3.0f,  3.0f,
    });

    topology topology(input_layout("input", input.get_layout()));

    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 16, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 16, 1, 1 } });

    set_values(weights,
        {
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f
        }
    );
    set_values(biases, { 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f});

    topology.add(
        data("weights", weights),
        data("bias", biases)
    );

    topology.add(
        convolution(
            "conv",
            "input",
            { "weights" },
            { "bias" },
            16,
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f, 8.0f, 8.0f, 3.65f, 3.65f,
        0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f, 0.5f, 0.5f, -5.36f, -5.36f,
        6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f, 6.0f, 6.0f, 3.65f, 3.65f,
        9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f, 9.0f, 9.0f, -5.36f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_group16_bfyx) {
    //  Test for grouped convolution, there are 16 joined weights and biases (group 16)
    //  data is similar as in basic_wsiz2x2_wstr2x2_in4x4x2x2_nopad_split2_depthwise_sep_opt_bfyx
    engine engine;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 16, 4, 4 } });

    set_values(input, {
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
        -0.5f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f, 0.0f, -1.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.5f, 2.0f, 1.5f, -0.5f,
        0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f, 0.5f, 1.5f, 2.3f, -0.4f, 2.0f, -4.0f, 1.0f, 3.0f,
    });

    topology topology(input_layout("input", input.get_layout()));

    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 16, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 16, 1, 1 } });

    set_values(weights,
        {
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f,
            -2.0f, 0.5f, 3.5f, 1.5f,
            -1.2f, 1.5f, 0.5f, -0.5f
        }
    );

    set_values(biases, { 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f, 2.0f, -1.0f});

    topology.add(
            data("weights", weights),
            data("bias", biases)
    );

    topology.add(
        convolution(
            "conv",
            "input",
            { "weights" },
            { "bias" },
            16,
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
        8.0f, 0.5f,  6.0f,  9.0f, 3.65f,-5.36f, 3.65f, -5.36f,
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  0.0
    //  f3: -0.5
    //
    //
    //  Filter1:
    //  -2 -0.5  ofm=0
    //   1  2    ofm=1
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  1.5  ofm=0
    //   2  0.5  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2.25
    //   7.5
    //
    //  -1.75
    //   2.25

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 4, 1, 1 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 2, 1, 1 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, {
       1.5f, 0.5f, 0.0f, -0.5f
    });
    set_values(weights1, { -2.0f, -0.5f, 1.0f, 2.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 1.5f, 2.0f, 0.5f });
    set_values(biases2, { -1.0f, 2.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-2.25f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(7.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(-1.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(2.25f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x2x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x2
    //  Output : 1x1x4
    //
    //  Input:
    //  f0:  1.5
    //
    //  f1:  0.5
    //
    //  Filter1:
    //  -2  ofm=0
    //   1  ofm=1
    //  Bias1:
    //   1  5
    //
    //  Filter2:
    //   4  ofm=0
    //   2  ofm=1
    //
    //  Bias2:
    //  -1  2.5
    //
    //  Output:
    //  -2
    //   6.5
    //
    //   1
    //   3.5

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 2, 1, 1 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 4 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 2, 1, 1, 1 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 2, 1, 1 } });

    set_values(input, {
        1.5f, 0.5f
    });
    set_values(weights1, { -2.0f, 1.0f });
    set_values(biases1, { 1.0f, 5.0f });
    set_values(weights2, { 4.0f, 2.0f });
    set_values(biases2, { -1.0f, 2.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(6.5f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(1.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(3.5f, get_value<float>(output_ptr, 3));
}

TEST(convolution_f32_fw_gpu, basic_wsiz1x1_wstr2x2_in1x1x4x1_filter_1x3x2x1x1_nopad_split2) {
    //  Filter : 1x1
    //  Stride : 2x2
    //  Input  : 1x1x4
    //  Output : 1x1x6
    //
    //  Input:
    //  f0:  1.5
    //  f1:  0.5
    //
    //  f2:  2
    //  f3: -1.0
    //
    //  Filter1:
    //  -2   1   ofm=0
    //   1   3   ofm=1
    //   0.5 8   ofm=2
    //  Bias1:
    //   1   5   3
    //
    //  Filter2:
    //   4  -4   ofm=0
    //   2   0.5 ofm=1
    //  -0.5 3   ofm=2
    //
    //  Bias2:
    //  -1   2.5 2
    //
    //  Output:
    //  -1.5
    //   8
    //   7.75
    //
    //   11
    //   6
    //  -2

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 4, 1, 1 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1,{ 1, 1 }, 6 } });
    auto weights1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 1 } });
    auto biases1 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 3, 1, 1 } });
    auto weights2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 3, 2, 1, 1 } });
    auto biases2 = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 3, 1, 1 } });

    set_values(input, {
        1.5f, 0.5f, 2.0f, -1.0f
    });
    set_values(weights1, { -2.0f, 1.0f, 1.0f, 3.0f, 0.5f, 8.0f });
    set_values(biases1, { 1.0f, 5.0f, 3.0f });
    set_values(weights2, { 4.0f, -4.0f, 2.0f, 0.5f, -0.5f, 3.0f });
    set_values(biases2, { -1.0f, 2.5f, 2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights1", weights1),
        data("biases1", biases1),
        data("weights2", weights2),
        data("biases2", biases2),
        convolution(
            "conv",
            "input",
            { "weights1", "weights2" },
            { "biases1", "biases2" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1,1,1,1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "conv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(-1.5f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(8.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(7.75f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(11.0f, get_value<float>(output_ptr, 3));
    EXPECT_FLOAT_EQ(6.0f, get_value<float>(output_ptr, 4));
    EXPECT_FLOAT_EQ(-2.0f, get_value<float>(output_ptr, 5));

}

TEST(convolution_gpu, trivial_convolution_relu) {

    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2

    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  -2
    //
    //  Output:
    //  4  0.0
    //  2  5

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1, 1, 1, 1 }),
        activation(
            "out",
            "conv",
            activation_func::relu
        )
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(0.0f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_ptr, 3));
}

TEST(convolution_gpu, relu_with_negative_slope) {

    //  Filter : 2x2
    //  Stride : 2x2
    //  Input  : 4x4
    //  Output : 2x2
    //  Negative Slope : 0.1

    //  Input:
    //  -0.5   1     0.5  2
    //   1.5  -0.5   0   -1
    //   0.5   0.5  -1    1
    //   0.5   2     1.5 -0.5
    //
    //  Filter
    //  -2   0.5
    //   3.5 1.5
    //
    //  Bias
    //  -2
    //
    //  Output:
    //  4  -0.35
    //  2  5

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::yxfb, { 1, 1, 4, 4 } });
    //auto output = memory::allocate({ memory::format::yxfb_f32,{ 1 ,{ 2, 2 }, 1 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 2, 2 } });
    auto biases = memory::allocate(engine, { data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        -0.5f,  1.0f,  0.5f,  2.0f,
        1.5f, -0.5f,  0.0f, -1.0f,
        0.5f,  0.5f, -1.0f,  1.0f,
        0.5f,  2.0f,  1.5f, -0.5f
    });
    set_values(weights, { -2.0f, 0.5f, 3.5f, 1.5f });
    set_values(biases, { -2.0f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,2,2 },
            { 0,0,0,0 },
            { 1, 1, 1, 1 }),
        activation(
            "out",
            "conv",
            activation_func::relu_negative_slope,
            {0.1f, 0.0f}
        )
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    EXPECT_FLOAT_EQ(4.0f, get_value<float>(output_ptr, 0));
    EXPECT_FLOAT_EQ(-0.35f, get_value<float>(output_ptr, 1));
    EXPECT_FLOAT_EQ(2.0f, get_value<float>(output_ptr, 2));
    EXPECT_FLOAT_EQ(5.0f, get_value<float>(output_ptr, 3));
}

TEST(convolution_gpu, DISABLED_two_1x1_kernels_after_each_other) {

    const auto& engine = get_test_engine();

    extern const std::vector<float> conv_1x1_output;

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 16, 8, 16, 16 } });
    auto weights_conv_1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 8, 8, 1, 1 } });
    auto weights_conv_2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 8, 1, 1 } });

    set_random_values<float>(input);
    set_random_values<float>(weights_conv_1);
    set_random_values<float>(weights_conv_2);

    auto inp_lay = input_layout("input", input.get_layout());
    auto conv_1 = convolution(
        "conv_1",
        "input",
        { "weights_conv_1" });
    auto conv_2 = convolution(
        "conv_2",
        "conv_1",
        { "weights_conv_2" });

    topology topology(
        inp_lay,
        data("weights_conv_1", weights_conv_1),
        conv_1,
        data("weights_conv_2", weights_conv_2),
        conv_2
    );

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));

    auto output_prim = outputs.at("conv_2").get_memory();

    auto output_ptr = output_prim.pointer<float>();
    auto output_layout = output_prim.get_layout();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    int f_offset = y_size * x_size;
    int b_offset = f_size * f_offset;
    for (int b = 0; b < b_size; ++b)
    {
        for (int f = 0; f < f_size; ++f)
        {
            for (int y = 0; y < y_size; ++y)
            {
                for (int x = 0; x < x_size; ++x)
                {
                    int idx = b * b_offset + f * f_offset + y * x_size + x;
                    EXPECT_TRUE(are_equal(conv_1x1_output[idx], get_value<float>(output_ptr, idx)));
                }
            }
        }
    }
}

TEST(convolution_gpu, basic_yxfb_4_4_yxfb_2_2_b16_if2_of16_st2_2_p0_sp1_fp32)
{
#define USE_OLD_WEIGHTS_FORMAT 0

    const auto input_format   = format::yxfb;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_format = format::bfyx;
#else
    const auto weights_format = format::yxfb;
#endif
    const auto biases_format = format::bfyx;

    const int32_t batch_size = 16;
    const int32_t input_feature_count = 2;
    const int32_t output_feature_count = 16;

    const int32_t stride_x = 2;
    const int32_t stride_y = 2;

    const int32_t input_x = 4;
    const int32_t input_y = 4;
    const int32_t weights_x = 2;
    const int32_t weights_y = 2;
    const int32_t output_x = (input_x - weights_x) / stride_x + 1;
    const int32_t output_y = (input_y - weights_y) / stride_y + 1;

    const auto& engine = get_test_engine();

    auto input_size = tensor( batch_size, input_feature_count, input_x, input_y );
    auto input = memory::allocate(engine, { data_types::f32, input_format, input_size });
    auto weights_size = tensor( output_feature_count, input_feature_count, weights_x, weights_y );
    auto weights = memory::allocate(engine, { data_types::f32, weights_format, weights_size });
    auto biases = memory::allocate(engine, { data_types::f32, biases_format, {1,output_feature_count,1,1}});

    //auto output = memory::allocate({output_format, {batch_size, {output_x, output_y}, output_feature_count}});

    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);

    // weights:
    std::vector<float> weights_vals_template {
        -4.0f, -2.0f,
         4.0f,  4.0f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
#if USE_OLD_WEIGHTS_FORMAT
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#else
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#endif
    set_values(weights, weights_vals);

    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);

    // output:
    std::vector<float> output_vals_template {
         9.0f, 10.0f,
        13.0f, 14.0f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }

    // Computing convolution.
    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(
            "conv",
            "input",
            { "weights" },
            { "biases" },
            { 1,1,stride_x,stride_y },
            { 0,0,0,0 },
            { 1, 1, 1, 1 }),
            activation(
                "out",
                "conv",
                activation_func::relu,
                { 0.1f, 0.0f }
            )
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i], get_value<float>(output_ptr, i));
                EXPECT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] << " != " << get_value<float>(output_ptr, i) << std::endl;
                    return;
                }
            }
        }
    }

#undef USE_OLD_WEIGHTS_FORMAT
}

template<typename T>
void quantize_weights(cldnn::memory& weights, cldnn::memory& w_qf)
{
    using std::abs;

    auto batch_pitch = weights.get_layout().get_pitches().batch[0];
    auto ptr = weights.pointer<T>();
    auto wqf_ptr = w_qf.pointer<float>();
    T max = (T) 0.0f;
    for (int ofm = 0; ofm < weights.get_layout().size.batch[0]; ofm++)
    {
        max = (T) 0.0f;
        for (int w = 0; w < batch_pitch; w++)
            if (max < abs(ptr[ofm* batch_pitch + w]))
                max = abs(ptr[ofm* batch_pitch + w]);

        if (max == (T)0)
            max = (T)1; // do not quantize

        for (int w = 0; w < batch_pitch; w++)
            ptr[ofm* batch_pitch + w] = (T)round((float)ptr[ofm* batch_pitch + w] * 127.0f / (float)max);
        wqf_ptr[ofm] = max/127.0f;
    }
}
template<typename T>
void calibrate(const cldnn::memory& output, cldnn::memory& calibrations)
{
    using std::abs;

    auto feature_pitch = output.get_layout().get_pitches().feature[0];
    auto ptr = output.pointer<T>();
    auto calibrations_ptr = calibrations.pointer<float>();
    T max = (T) 0.0f;
    for (int ofm = 0; ofm < output.get_layout().size.feature[0]; ofm++)
    {
        max = (T) 0.0f;
        for (int w = 0; w < feature_pitch; w++)
            if (max < abs(ptr[ofm* feature_pitch + w]))
                max = abs(ptr[ofm* feature_pitch + w]);
        calibrations_ptr[ofm] =  127.0f / max;
    }
}

template<typename T>
T max_abs(const cldnn::memory& mem)
{
    using std::abs;

    T max = (T)0;
    auto ptr = mem.pointer<T>();
    for (auto& a : ptr)
        if (max < abs(a))
            max = abs(a);
    return max;
}

template<typename T>
void apply_calibration_on_weights(cldnn::memory& weights, cldnn::memory& qf)
{
    auto ptr = weights.pointer<T>();
    auto wqf_ptr = qf.pointer<float>();
    tensor w_size = weights.get_layout().size;
    int index = 0;
    for (int ofm = 0; ofm < w_size.batch[0]; ofm++)
        for (int ifm = 0; ifm < w_size.feature[0]; ifm++)
            for (int xy = 0; xy < w_size.spatial[0] * w_size.spatial[1]; xy++)
            {
                ptr[index] = ptr[index] / wqf_ptr[ifm];
                index++;
            }
}

cldnn::memory create_int8_weights(engine engine, cldnn::memory& in_weights)
{
    auto layout = in_weights.get_layout();
    auto out_weights = memory::allocate(engine, { data_types::i8, layout.format, layout.size });
    auto in = in_weights.pointer<float>();
    auto out = out_weights.pointer<char>();
    int indx = 0;
    for (auto& a : in)
        out[indx++] = (char) a;
    return out_weights;
}

void add_primitives(const engine& engine, topology& topology)
{
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 1, 3, 2 } });

    std::vector<char> weights_values = { 1, 2, 1,
                                         2, 1, 2,

                                         19, 17, -1,
                                         -10, 32, 23 };
    set_values<char>(weights, weights_values);
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    set_values(biases, { 1.0f, -8.0f });

    topology.add(
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, { 0, 0, 1, 2 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 }),
        activation( "out", "conv", activation_func::relu)
    );
}

TEST(convolution_f32_fw_gpu, byte_activation) {
    //  Filter : 2x3
    //  Stride : 2x1
    //  Input  : 4x5
    //  Output : 2x3
    //
    //  Input:
    //  1  2  3  4  5
    //  2  2  3  4  6
    //  3  3  3  5  1
    //  1  1  1  1  1
    //
    //  Filter:
    //  1  2  1
    //  2  1  2
    //
    //  19 17 -1
    // -10 32 23
    //
    //  Output:
    // 21  28  39
    // 18  20  20
    //
    // -101 -11 92
    // -114 -116 -78
    //
    //  Bias:
    //  1 -8
    engine_configuration eng_conf(false, false, false, "", "", true, "", "/home/vparamuz/tmp/cldnn/sources/");
    engine engine{ eng_conf };
    auto input = memory::allocate(engine, { data_types::i8, format::bfyx,{ 1, 1, 5, 4 } });

    VVVF<char> output_vec = {
        {
            { 11, 0, 15 },
            { 0,  0, 2 }
        },
        {
            { 33, 0, 0 },
            { 0, 0, 0 }
        } };

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    opts.set_option(build_option::graph_dumps_dir("graph"));

    set_values<char>(input, {  1,  2, -3,  4, -5,
                               2, -2,  3, -4,  6,
                              -3,  3, -3,  5, -1,
                              -1, -1, -1, -1, -1 });

    topology topology(
        input_layout("input", input.get_layout()));
    add_primitives(engine, topology);
    network network(engine, topology, opts);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_layout = output_memory.get_layout();
    auto output_ptr = output_memory.pointer<float>();

    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_NEAR(output_vec[f][y][x], ((float)output_ptr[f*y_size*x_size + y * x_size + x]), 3.0f);
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_symmetric) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 1, 3, 3 } });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 52.0f, 78.0f, 3.0f },
            { 8.0f, 14.0f, 0.0f }
        },
        {
            { 20.0f, 35.0f, 31.0f },
            { 11.0f, 19.0f, 0.0f }
        } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution("conv", "input", { "weights" }, { "biases" }, tensor{ 0, 0, 2, 2 }, tensor(0), tensor{1, 1, 1, 1}, tensor{1, 2, 3, 2}),
        reorder("out", "conv", format::bfyx, data_types::f32));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network(engine, topology, opts);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_ptr = output_memory.pointer<float>();

    auto output_layout = output_memory.get_layout();
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_NEAR(output_vec[f][y][x], ((float)output_ptr[f*y_size*x_size + y * x_size + x]), 1e-5f) <<
                " x="<<x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_weight_and_activations) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 1, 3, 3 } });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto w_zp = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 1, 1, 1 } });
    auto a_zp = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2 });
    set_values<int8_t>(w_zp, { 1, -1 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 12.0f, 26.0f, -19.0f },
            { 2.0f, 8.0f, 4.0f }
        },
        {
            { -8.0f, 19.0f, 21.0f },
            { -7.0f, 1.0f, -18.0f }
        } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        data("w_zp", w_zp),
        convolution("conv", "input", { "weights" }, { "biases" }, { "w_zp" }, { "a_zp" }, 1, data_types::f32,
                    tensor{ 0, 0, 2, 2 }, tensor(0), tensor{1, 1, 1, 1}, tensor{1, 2, 3, 2}),
        reorder("out", "conv", format::bfyx, data_types::f32));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network(engine, topology, opts);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_ptr = output_memory.pointer<float>();

    auto output_layout = output_memory.get_layout();
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_NEAR(output_vec[f][y][x], ((float)output_ptr[f*y_size*x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_activations_per_tensor) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 1, 3, 3 } });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto a_zp = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 16.0f, 42.0f, -13.0f },
            { 2.0f, 8.0f, 2.0f }
        },
        {
            { -12.0f, 3.0f, 15.0f },
            { -7.0f, 1.0f, -16.0f }
        } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        convolution("conv", "input", { "weights" }, { "biases" }, { }, { "a_zp" }, 1, data_types::f32,
                    tensor{ 0, 0, 2, 2 }, tensor(0), tensor{1, 1, 1, 1}, tensor{1, 2, 3, 2}),
        reorder("out", "conv", format::bfyx, data_types::f32));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network(engine, topology, opts);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_ptr = output_memory.pointer<float>();

    auto output_layout = output_memory.get_layout();
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_NEAR(output_vec[f][y][x], ((float)output_ptr[f*y_size*x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_activations_per_channel) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 2, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 2, 3, 3 } });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto a_zp = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 2, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1,

                                 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });

    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5,

                                   1, 2, -1,
                                   -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                   -1, 3,  2,
                                   0, 2,  5 });
    set_values<uint8_t>(a_zp, { 2, 5 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { -36.0f, 5.0f, -14.0f },
            { -24.0f, -10.0f, -30.0f }
        },
        {
            { -45.0f, -4.0f, -23.0f },
            { -33.0f, -19.0f, -39.0f }
        } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        convolution("conv", "input", { "weights" }, { "biases" }, { }, { "a_zp" }, 1, data_types::f32,
                    tensor{ 0, 0, 2, 2 }, tensor(0), tensor{1, 1, 1, 1}, tensor{1, 2, 3, 2}),
        reorder("out", "conv", format::bfyx, data_types::f32));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network(engine, topology, opts);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_ptr = output_memory.pointer<float>();

    auto output_layout = output_memory.get_layout();
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_NEAR(output_vec[f][y][x], ((float)output_ptr[f*y_size*x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_activations_per_channel_3ic_with_sub) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 3, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 3, 3, 3 } });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto a_zp = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 3, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1,

                                 2, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1,

                                 3, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });

    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5,

                                   1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });

    set_values<uint8_t>(a_zp, { 2, 5, 6 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { -82.0f, -26.0f, -60.0f },
            { -35.0f, -15.0f, -25.0f }
        },
        {
            { -86.0f, -57.0f, -32.0f },
            { -68.0f, -46.0f, -79.0f }
        } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("a_zp", a_zp),
        activation("activation", "input", activation_func::relu),  // needed just to add padding
        eltwise("in", {"activation", "a_zp"}, eltwise_mode::sub, data_types::f32),
        convolution("conv", "in", { "weights" }, { "biases" }, 1,
                    tensor{ 0, 0, 2, 2 }, tensor(0), tensor{1, 1, 1, 1}, tensor{1, 2, 3, 2}, data_types::f32),
        reorder("out", "conv", format::bfyx, data_types::f32));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network(engine, topology, opts);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_ptr = output_memory.pointer<float>();

    auto output_layout = output_memory.get_layout();
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_NEAR(output_vec[f][y][x], ((float)output_ptr[f*y_size*x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

TEST(convolution_int8_fw_gpu, quantized_convolution_u8s8f32_asymmetric_weights_per_channel) {
    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::u8, format::bfyx,{ 1, 1, 5, 4 } });
    auto weights = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 1, 3, 3 } });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 1, 1 } });
    auto w_zp = memory::allocate(engine, { data_types::i8, format::bfyx,{ 2, 1, 1, 1 } });

    set_values<uint8_t>(input, { 1, 2, 3, 4, 5,
                                 2, 2, 3, 4, 6,
                                 3, 3, 3, 5, 1,
                                 1, 1, 1, 1, 1 });
    set_values<int8_t>(weights, {  1, 2, -1,
                                  -2, 1,  2,
                                   9, 7, -1,

                                   9, 0, -4,
                                  -1, 3,  2,
                                   0, 2,  5 });
    set_values<int8_t>(w_zp, { 1, -1 });
    set_values(biases, { 1.0f, -8.0f });

    VVVF<float> output_vec = {
        {
            { 30.0f, 44.0f, -9.0f },
            { -4.0f, 2.0f, -2.0f }
        },
        {
            { 42.0f, 69.0f, 43.0f },
            { 23.0f, 31.0f, 2.0f }
        } };

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("biases", biases),
        data("w_zp", w_zp),
        convolution("conv", "input", { "weights" }, { "biases" }, { "w_zp" }, { }, 1, data_types::f32,
                    tensor{ 0, 0, 2, 2 }, tensor(0), tensor{1, 1, 1, 1}, tensor{1, 2, 3, 2}),
        reorder("out", "conv", format::bfyx, data_types::f32));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network(engine, topology, opts);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output_memory = outputs.at("out").get_memory();
    auto output_ptr = output_memory.pointer<float>();

    auto output_layout = output_memory.get_layout();
    int y_size = output_layout.size.spatial[1];
    int x_size = output_layout.size.spatial[0];
    int f_size = output_layout.size.feature[0];
    int b_size = output_layout.size.batch[0];
    EXPECT_EQ(output_layout.format, format::bfyx);
    EXPECT_EQ(y_size, 2);
    EXPECT_EQ(x_size, 3);
    EXPECT_EQ(f_size, 2);
    EXPECT_EQ(b_size, 1);
    for (int f = 0; f < f_size; f++)
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                EXPECT_NEAR(output_vec[f][y][x], ((float)output_ptr[f*y_size*x_size + y * x_size + x]), 1e-5f) <<
                " x="<< x << " y=" << y << " f=" << f;
            }
        }
}

template <typename InputTy, typename OutputTy, typename PreActivationTy = int32_t>
class ConvQuantizedTest : public testing::Test {
protected:
    std::vector<InputTy> input_values;
    // As for the depthwise convolution, will be processed to a normal
    // convolution later on.
    std::vector<int8_t> weights_values;
    std::vector<int32_t> biases_values;
    std::vector<float> quantization_values;
    std::vector<PreActivationTy> output_pre_relu; // ...but after quantization.

    void add_feature(std::vector<InputTy> input,
                     std::vector<int8_t> weights,
                     int32_t bias,
                     float quantization,
                     std::vector<PreActivationTy> output)
    {
        input_values.insert(input_values.end(), input.begin(), input.end());
        weights_values.insert(
            weights_values.end(), weights.begin(), weights.end());
        biases_values.push_back(bias);
        quantization_values.push_back(quantization);
        output_pre_relu.insert(
            output_pre_relu.end(), output.begin(), output.end());
    }

    template<typename T = PreActivationTy>
    static typename std::enable_if<std::is_floating_point<T>::value>::type
    expect_eq(const PreActivationTy& lhs, const PreActivationTy& rhs)
    {
        EXPECT_NEAR(lhs, rhs, 0.001f);
    }

    template<typename T = PreActivationTy>
    static typename std::enable_if<std::is_integral<T>::value>::type
    expect_eq(const PreActivationTy& lhs, const PreActivationTy& rhs)
    {
        EXPECT_EQ(lhs, rhs);
    }

    template <typename T>
    static T pre_relu_to_output(T pre_relu) {
      // No std::clamp before C++17 :(
      return std::min(
          static_cast<T>(std::numeric_limits<OutputTy>::max()),
          std::max(static_cast<T>(std::numeric_limits<OutputTy>::lowest()),
                   std::max(static_cast<T>(0), pre_relu)));
    }

    void do_test()
    {
        const auto& engine = get_test_engine();
        int n_features = static_cast<int>(biases_values.size());

        auto input_shape = tensor(1, n_features, 4, 1);
        auto weights_shape = tensor(n_features, n_features, 3, 1);
        auto biases_shape = tensor(1, n_features, 1, 1);

        auto input = memory::allocate(
            engine,
            {type_to_data_type<InputTy>::value, format::bfyx, input_shape});
        auto weights = memory::allocate(
            engine, {data_types::i8, format::bfyx, weights_shape});

        auto biases = memory::allocate(
            engine, {data_types::i32, format::bfyx, biases_shape});
        auto quantization = memory::allocate(
            engine, {data_types::f32, format::bfyx, biases_shape});

        set_values(input, input_values);
        std::vector<int8_t> post_processed_weights_values(n_features
                                                          * n_features * 3);
        for (int output_feature = 0; output_feature < n_features; ++output_feature)
            for (int input_feature = 0; input_feature < n_features;
                 ++input_feature)
                for (int x = 0; x < 3; ++x)
                {
                    int idx =
                        output_feature * n_features * 3 + input_feature * 3 + x;
                    if (input_feature == output_feature)
                        post_processed_weights_values[idx] =
                            weights_values[input_feature * 3 + x];
                    else
                        post_processed_weights_values[idx] = 0;
                }
        set_values(weights, post_processed_weights_values);
        set_values(biases, biases_values);
        set_values(quantization, quantization_values);

        build_options opts;
        opts.set_option(build_option::optimize_data(false));
        opts.set_option(build_option::graph_dumps_dir("/tmp/cldnn_dumps/"));

        topology topology(input_layout("input", input.get_layout()),
                          data("weights", weights),
                          data("biases", biases),
                          data("quantization", quantization),
                          convolution("conv",
                                      "input",
                                      {"weights"},
                                      {"biases"},
                                      {1, 1, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1, 1, 1}),
            activation("out", "conv", activation_func::relu));

        network network(engine, topology, opts);
        network.set_input_data("input", input);

        auto outputs = network.execute();

        auto output_memory = outputs.at("out").get_memory();
        auto output_layout = output_memory.get_layout();
        auto output_ptr = output_memory.pointer<OutputTy>();
        int y_size = output_layout.size.spatial[1];
        int x_size = output_layout.size.spatial[0];
        int f_size = output_layout.size.feature[0];
        int b_size = output_layout.size.batch[0];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(y_size, 1);
        EXPECT_EQ(x_size, 2);
        EXPECT_EQ(f_size, n_features);
        EXPECT_EQ(b_size, 1);

        for (int f = 0; f < f_size; f++)
            for (int x = 0; x < x_size; ++x)
            {
                // printf("f: %d, x: %d\n", f, x);
                PreActivationTy expected = pre_relu_to_output(output_pre_relu[f * x_size + x]);
                auto actual = static_cast<PreActivationTy>(output_ptr[f * x_size + x]);
                expect_eq(expected, actual);
            }
    }
};

class ConvQuantizedTest_i8_to_u8 : public ConvQuantizedTest<int8_t, uint8_t>
{};

TEST_F(ConvQuantizedTest_i8_to_u8, DISABLED_basic) {
    // Check that the output precision is `u8` indeed.
    add_feature({125, 125, 0, 1}, {2, 0, 1}, 1, 1.0f, {251, 252});

    // Check ReLU (negative result will become zero in the output).
    add_feature({0, 50, 0, -50}, {0, 4, 4}, 1, 1.0f, {201, -199});

    // Same but with non-unit calibration (just in case).
    add_feature({0, 50, 0, -50}, {0, 8, 8}, 2, 0.5f, {201, -199});

    // Something with intermediate accumulator outside i8/u8 range.
    add_feature({120, 120, 120, -120}, {1, 1, 1}, 0, 0.25f, {90, 30});

    // Check rounding (TODO: currently rounding to nearest, with half rounded
    // away from zero, might need to change that).
    add_feature({125, 125, 0, 126}, {1, 1, 1}, 1, 0.5f, {126, 126});
    add_feature({125, 125, 0, 126}, {1, 1, 1}, 2, 0.5f, {126, 127});
    // Same, but with output outside the i8 range.
    add_feature({125, 125, 0, 126}, {1, 1, 1}, 21, 0.5f, {136, 136});

    // Check saturation.
    add_feature({0, 50, 0, -50}, {0, 8, 8}, 2, 1.0f, {402, -398});

    do_test();
}

class ConvQuantizedTest_u8_to_u8 : public ConvQuantizedTest<uint8_t, uint8_t>
{};

TEST_F(ConvQuantizedTest_u8_to_u8, DISABLED_basic) {
    // Start with the tests from the "i8" input case (move negative sign to the
    // weights were needed):
    //
    // Check that the output precision is `u8` indeed.
    add_feature({125, 125, 0, 1}, {2, 0, 1}, 1, 1.0f, {251, 252});

    // Check ReLU (negative result will become zero in the output).
    add_feature({0, 50, 0, 50}, {0, 4, -4}, 1, 1.0f, {201, -199});

    // Same but with non-unit calibration (just in case).
    add_feature({0, 50, 0, 50}, {0, 8, -8}, 2, 0.5f, {201, -199});

    // Something with intermediate accumulator outside i8/u8 range.
    add_feature({240, 240, 240, 240}, {2, 1, -1}, 0, 0.125f, {60, 60});

    // Check rounding (TODO: currently rounding to nearest, with half rounded
    // away from zero, might need to change that).
    add_feature({125, 125, 0, 126}, {1, 1, 1}, 1, 0.5f, {126, 126});
    add_feature({125, 125, 0, 126}, {1, 1, 1}, 2, 0.5f, {126, 127});
    // Same, but with output outside the i8 range.
    add_feature({125, 125, 0, 126}, {1, 1, 1}, 21, 0.5f, {136, 136});

    // Check saturation.
    add_feature({0, 50, 0, 50}, {0, 8, -8}, 2, 1.0f, {402, -398});

    // Now, something "u8"-input-specific (basically subset of the tests above
    // but move the scaling from the weights to the input):
    add_feature({250, 250, 0, 1}, {1, 0, 1}, 1, 1.0f, {251, 252});
    add_feature({0, 200, 0, 200}, {0, 1, -1}, 1, 1.0f, {201, -199});
    add_feature({0, 200, 0, 200}, {0, 2, -2}, 2, 0.5f, {201, -199});

    do_test();
}

class ConvQuantizedTest_u8_to_i8 : public ConvQuantizedTest<uint8_t, int8_t>
{};

TEST_F(ConvQuantizedTest_u8_to_i8, DISABLED_basic) {
    // Basic test + rounding
    add_feature({125, 125, 0, 1}, {2, 0, 1}, 1, 0.5f, {126, 126});

    // Test proper clamping to the output i8 range.
    add_feature({125, 125, 0, 1}, {2, 0, 1}, 1, 1.0f, {251, 252});

    // Test ReLU by having negative number pre-ReLU.
    add_feature({0, 50, 0, 50}, {0, 1, -1}, 1, 1.0f, {51, -49});

    do_test();
}

class ConvQuantizedTest_i8_to_float : public ConvQuantizedTest<int8_t, float, float>
{};

TEST_F(ConvQuantizedTest_i8_to_float, DISABLED_basic) {
    // Some basic checks.
    add_feature({125, 125, 0, 1}, {2, 0, 1}, 1, 1.0f, {251.0f, 252.0f});
    add_feature({0, 50, 0, -50}, {0, 8, 8}, 2, 0.5f, {201.0f, -199.0f});
    add_feature({0, 50, 0, -50}, {0, 8, 8}, 2, 1.0f, {402.0f, -398.0f});

    // Check the FP accuracy - no rounding should be performed.
    add_feature({0, 5, 0, -5}, {0, 8, 8}, 0, 1.01f, {40.4f, -40.4f});

    do_test();
}

TEST(convolution_f32_fw_gpu, local_basic) {
    //  Filter : 3x3x2x2 - local sizes
    //  Stride : 1x1
    //  Input  : 4x4
    //  Output : 3x3
    //
    //  Input:
    //  1  1  1  1
    //  1  1  1  1
    //  2  2  2  2
    //  2  2  2  2
    //
    //
    //  Filter:
    //  0 0  1 1  2 2
    //  0 0  1 1  2 2
    //
    //  3 3  4 4  5 5
    //  3 3  4 4  5 5
    //
    //  6 6  7 7  8 8
    //  6 6  7 7  8 8
    //
    //  Output:
    //  0  4  8
    // 18 24 30
    // 48 56 64
    //

    const auto& engine = get_test_engine();
    tensor local_size = tensor(1,1,2,2,3,3);
    auto input_f = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 4 } });
    auto weights_f = memory::allocate(engine, { data_types::f32, format::bf_lyx_yx, local_size });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    std::vector<float> weights_values_f = {
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,

        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0,
        5.0, 5.0, 5.0, 5.0,

        6.0, 6.0, 6.0, 6.0,
        7.0, 7.0, 7.0, 7.0,
        8.0, 8.0, 8.0, 8.0,
    };
    set_values<float>(input_f, { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 });
    set_values<float>(weights_f, weights_values_f);
    set_values(biases, { 0.0f });
    std::vector<float> output_vec =
    {
        0.0f, 4.0f, 8.0f,
        18.0f, 24.0f, 30.0f,
        48.0f, 56.0f, 64.0f
    };

    topology topology_f(
        input_layout("input_f", input_f.get_layout()),
        data("weights_f", weights_f),
        data("biases", biases),
        convolution("conv_f", "input_f", { "weights_f" }, { "biases" }, { 0, 0, 1, 1 }));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network_f(engine, topology_f, opts);
    network_f.set_input_data("input_f", input_f);

    auto outputs_f = network_f.execute();
    EXPECT_EQ(outputs_f.begin()->first, "conv_f");

    auto output_memory_f = outputs_f.at("conv_f").get_memory();
    auto output_ptr_f = output_memory_f.pointer<float>();
    unsigned int cntr = 0;
    for (auto fl : output_ptr_f)
        EXPECT_FLOAT_EQ(fl, output_vec[cntr++]);
}

TEST(convolution_f32_fw_gpu, local_multi_out_features) {
    //  Filter : 3x1x3x3x2x2 - local sizes
    //  Stride : 1x1
    //  Input  : 4x4
    //  Output : 3x3x3
    //
    //  Input:
    //  1  1  1  1
    //  1  1  1  1
    //  2  2  2  2
    //  2  2  2  2
    //
    //
    //  Filter:
    //  0 0  1 1  2 2  --- 1 ofm
    //  0 0  1 1  2 2
    //
    //  3 3  4 4  5 5
    //  3 3  4 4  5 5
    //
    //  6 6  7 7  8 8
    //  6 6  7 7  8 8
    //
    //  0 0  0 0  0 0  --- 2 ofm
    //  0 0  0 0  0 0
    //
    //  0 0  0 0  0 0
    //  0 0  0 0  0 0
    //
    //  0 0  0 0  0 0
    //  0 0  0 0  0 0
    //
    //  0 0  2 2  4 4 --- 3 ofm
    //  0 0  2 2  4 4
    //
    //  6 6  8 8  1 1
    //  6 6  8 8  1 1
    //
    //  3 3  5 5  7 7
    //  3 3  5 5  7 7
    //

    //
    //  Output:
    //  0  4  8
    // 18 24 30
    // 48 56 64
    //
    //  0  0  0
    //  0  0  0
    //  0  0  0
    //
    //  0  8 16
    // 36 48  6
    // 24 40 56
    //

    const auto& engine = get_test_engine();
    tensor local_size = tensor(3,1,2,2,3,3);
    auto input_f = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 4 } });
    auto weights_f = memory::allocate(engine, { data_types::f32, format::bf_lyx_yx, local_size });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 3, 1, 1 } });

    std::vector<float> weights_values_f = {
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,

        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0,
        5.0, 5.0, 5.0, 5.0,

        6.0, 6.0, 6.0, 6.0,
        7.0, 7.0, 7.0, 7.0,
        8.0, 8.0, 8.0, 8.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 0.0,
        2.0, 2.0, 2.0, 2.0,
        4.0, 4.0, 4.0, 4.0,

        6.0, 6.0, 6.0, 6.0,
        8.0, 8.0, 8.0, 8.0,
        1.0, 1.0, 1.0, 1.0,

        3.0, 3.0, 3.0, 3.0,
        5.0, 5.0, 5.0, 5.0,
        7.0, 7.0, 7.0, 7.0,
    };
    set_values<float>(input_f, { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 });
    set_values<float>(weights_f, weights_values_f);
    set_values(biases, { 0.0f, 0.0f, 0.0f });
    std::vector<float> output_vec =
    {
        0.0f,  4.0f,  8.0f,
        18.0f, 24.0f, 30.0f,
        48.0f, 56.0f, 64.0f,

        0.0f,  0.0f, 0.0f,
        0.0f,  0.0f, 0.0f,
        0.0f,  0.0f, 0.0f,

        0.0f,  8.0f, 16.0f,
        36.0f, 48.0f,  6.0f,
        24.0f, 40.0f, 56.0f,
    };

    topology topology_f(
        input_layout("input_f", input_f.get_layout()),
        data("weights_f", weights_f),
        data("biases", biases),
        convolution("conv_f", "input_f", { "weights_f" }, { "biases" }, { 0, 0, 1, 1 }));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network_f(engine, topology_f, opts);
    network_f.set_input_data("input_f", input_f);

    auto outputs_f = network_f.execute();
    EXPECT_EQ(outputs_f.begin()->first, "conv_f");

    auto output_memory_f = outputs_f.at("conv_f").get_memory();
    auto output_ptr_f = output_memory_f.pointer<float>();
    unsigned int cntr = 0;
    for (auto fl : output_ptr_f)
    {
        EXPECT_FLOAT_EQ(fl, output_vec[cntr++]);
    }
}

TEST(convolution_f32_fw_gpu, local_multi_input_features) {
    //  Filter : 1x3x3x3x2x2 - local sizes
    //  Stride : 1x1
    //  Input  : 3x4x4
    //  Output : 3x3
    //
    //  Input:
    //  0  0  0  0
    //  0  0  0  0
    //  0  0  0  0
    //  0  0  0  0
    //
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //  1  1  1  1
    //
    //  2  2  2  2
    //  2  2  2  2
    //  2  2  2  2
    //  2  2  2  2
    //
    //
    //  Filter:
    //  0 0  1 1  2 2  <-- IFM 0
    //  0 0  1 1  2 2
    //
    //  3 3  4 4  5 5
    //  3 3  4 4  5 5
    //
    //  6 6  7 7  8 8
    //  6 6  7 7  8 8
    //
    //  0 0  1 1  2 2 <-- IFM 1
    //  0 0  1 1  2 2
    //
    //  3 3  4 4  5 5
    //  3 3  4 4  5 5
    //
    //  6 6  7 7  8 8
    //  6 6  7 7  8 8
    //
    //  0 0  1 1  2 2 <-- IFM 2
    //  0 0  1 1  2 2
    //
    //  3 3  4 4  5 5
    //  3 3  4 4  5 5
    //
    //  6 6  7 7  8 8
    //  6 6  7 7  8 8
    //
    //  Output:
    //  0 12 24
    // 36 48 60
    // 72 84 96
    //

    const auto& engine = get_test_engine();
    tensor local_size = tensor(1,3,2,2,3,3);
    auto input_f = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 3, 4, 4 } });
    auto weights_f = memory::allocate(engine, { data_types::f32, format::bf_lyx_yx, local_size });
    cldnn::memory biases = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    std::vector<float> weights_values_f = {
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,

        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0,
        5.0, 5.0, 5.0, 5.0,

        6.0, 6.0, 6.0, 6.0,
        7.0, 7.0, 7.0, 7.0,
        8.0, 8.0, 8.0, 8.0,

        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,

        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0,
        5.0, 5.0, 5.0, 5.0,

        6.0, 6.0, 6.0, 6.0,
        7.0, 7.0, 7.0, 7.0,
        8.0, 8.0, 8.0, 8.0,

        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,

        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0,
        5.0, 5.0, 5.0, 5.0,

        6.0, 6.0, 6.0, 6.0,
        7.0, 7.0, 7.0, 7.0,
        8.0, 8.0, 8.0, 8.0,
    };
    set_values<float>(input_f, {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 });
    set_values<float>(weights_f, weights_values_f);
    set_values(biases, { 0.0f });
    std::vector<float> output_vec =
    {
         0.0f, 12.0f, 24.0f,
        36.0f, 48.0f, 60.0f,
        72.0f, 84.0f, 96.0f
    };

    topology topology_f(
        input_layout("input_f", input_f.get_layout()),
        data("weights_f", weights_f),
        data("biases", biases),
        convolution("conv_f", "input_f", { "weights_f" }, { "biases" }, { 0, 0, 1, 1 }));

    build_options opts;
    opts.set_option(build_option::optimize_data(true));
    network network_f(engine, topology_f, opts);
    network_f.set_input_data("input_f", input_f);

    auto outputs_f = network_f.execute();
    EXPECT_EQ(outputs_f.begin()->first, "conv_f");

    auto output_memory_f = outputs_f.at("conv_f").get_memory();
    auto output_ptr_f = output_memory_f.pointer<float>();
    unsigned int cntr = 0;
    for (auto fl : output_ptr_f)
        EXPECT_FLOAT_EQ(fl, output_vec[cntr++]);
}

TEST(convolution_gpu, basic_yxfb_4_4_yxfb_2_2_b16_if2_of16_st2_2_p0_sp1_fp16)
{
#define USE_OLD_WEIGHTS_FORMAT 0

    const auto& engine = get_test_engine();

    if (!engine.get_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    const auto input_format   = format::yxfb;
#if USE_OLD_WEIGHTS_FORMAT
    const auto weights_format = format::bfyx;
#else
    const auto weights_format = format::yxfb;
#endif
    const auto biases_format  = format::bfyx;
    const auto output_format  = input_format;

    const int32_t batch_size = 16;
    const int32_t input_feature_count = 2;
    const int32_t output_feature_count = 16;

    const int32_t stride_x = 2;
    const int32_t stride_y = 2;

    const int32_t input_x = 4;
    const int32_t input_y = 4;
    const int32_t weights_x = 2;
    const int32_t weights_y = 2;
    const int32_t output_x = (input_x - weights_x) / stride_x + 1;
    const int32_t output_y = (input_y - weights_y) / stride_y + 1;

    auto input_size = tensor( batch_size, input_feature_count, input_x, input_y );
    auto input = memory::allocate(engine, { data_types::f32, input_format, input_size });
    auto weights_size = tensor( output_feature_count, input_feature_count, weights_x, weights_y );
    auto weights = memory::allocate(engine, { data_types::f32, weights_format, weights_size });
    auto biases_size = tensor( 1,output_feature_count,1,1 );
    auto biases = memory::allocate(engine, { data_types::f32, biases_format, biases_size });
    auto output_size = tensor( batch_size, output_feature_count, output_x, output_y );
    //auto output = memory::allocate({output_format, {batch_size, {output_x, output_y}, output_feature_count}});

    //auto input_cvtd = memory::allocate(engine, { data_types::f16, input_size });
    //auto weights_cvtd = memory::allocate(engine, { data_types::f16, weights_size });
    //auto biases_cvtd = memory::allocate(engine, { data_types::f16, biases_size });
    //auto output_cvtd  = memory::allocate({output_cvt_format, {batch_size, {output_x, output_y}, output_feature_count}});

    // input:
    std::vector<float> input_vals_template {
        0.25f, 0.50f, 0.75f, 1.00f,
        1.25f, 1.50f, 1.75f, 2.00f,
        2.25f, 2.50f, 2.75f, 3.00f,
        3.25f, 3.50f, 3.75f, 4.00f,
    };
    input_vals_template.resize(input_y * input_x);

    std::vector<float> input_vals;
    input_vals.reserve(input_y * input_x * input_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < input_y * input_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                input_vals.push_back((bi * input_feature_count + ifi + 1) * input_vals_template[yxi]);
            }
        }
    }
    set_values(input, input_vals);

    // weights:
    std::vector<float> weights_vals_template {
        -0.50f, -0.25f,
         0.50f,  0.50f,
    };
    weights_vals_template.resize(weights_y * weights_x);

    std::vector<float> weights_vals;
    weights_vals.reserve(weights_y * weights_x * input_feature_count * output_feature_count);
#if USE_OLD_WEIGHTS_FORMAT
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#else
    for (uint32_t yxi = 0; yxi < weights_y * weights_x; ++yxi)
    {
        for (uint32_t ifi = 0; ifi < input_feature_count; ++ifi)
        {
            for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
            {
                weights_vals.push_back((ofi * input_feature_count + ifi + 1) * weights_vals_template[yxi]);
            }
        }
    }
#endif
    set_values(weights, weights_vals);

    // biases:
    std::vector<float> biases_vals;
    biases_vals.reserve(output_feature_count);
    for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
    {
        biases_vals.push_back(ofi * 1.0f);
    }
    set_values(biases, biases_vals);

    // output:
    std::vector<float> output_vals_template {
        1.125f,  1.250f,
        1.625f,  1.750f,
    };
    output_vals_template.resize(output_y * output_x);

    std::vector<float> output_vals;
    output_vals.reserve(output_y * output_x * output_feature_count * batch_size);
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi)
            {
                uint32_t template_factor = input_feature_count * input_feature_count * input_feature_count * bi * ofi +
                    input_feature_count * input_feature_count * (input_feature_count + 1) / 2 * (bi + ofi) +
                    input_feature_count * (input_feature_count + 1) * (2 * input_feature_count + 1) / 6;
                float bias_factor = ofi * 1.0f;

                output_vals.push_back(template_factor * output_vals_template[yxi] + bias_factor);
            }
        }
    }

    //auto expected_float = memory::allocate(engine, { data_types::f32,{ format::x,{ static_cast<int32_t>(output_vals.size()) } } });
    //auto expected_half  = memory::allocate(engine, { data_types::f16,{ format::x,{ static_cast<int32_t>(output_vals.size()) } } });
    //auto expected       = memory::allocate(engine, { data_types::f32,{ format::x,{ static_cast<int32_t>(output_vals.size()) } } });

//    set_values(expected_float, output_vals);
//    auto cvt_expected_f32_f16 = reorder::create({expected_float, expected_half});
//    auto cvt_expected_f16_f32 = reorder::create({expected_half, expected});
//    execute({cvt_expected_f32_f16, cvt_expected_f16_f32}).wait();
//
//    auto expected_ptr = expected.as<const memory&>().pointer<float>();

    // Computing convolution.
    topology topology(
        input_layout("input", input.get_layout()),
        reorder("cvt_input", "input", {data_types::f16, input_format, input_size}),
        data("weights", weights),
        reorder("cvt_weights", "weights", {data_types::f16, weights_format, weights_size}),
        data("biases", biases),
        reorder("cvt_biases", "biases", {data_types::f16, biases_format, biases_size}),
        convolution(
            "conv",
            "cvt_input",
            { "cvt_weights" },
            { "cvt_biases" },
            { 1,1,stride_x,stride_y }),
        reorder("output", "conv", {data_types::f32, output_format, output_size})
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "output");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    // Checking result.
    uint32_t i = 0;
    for (uint32_t yxi = 0; yxi < output_y * output_x; ++yxi)
    {
        for (uint32_t ofi = 0; ofi < output_feature_count; ++ofi)
        {
            for (uint32_t bi = 0; bi < batch_size; ++bi, ++i)
            {
                auto equal = are_equal(output_vals[i] /*get_value(expected_ptr, i)*/, output_ptr[i], 0.002f);
                EXPECT_TRUE(equal);
                if (!equal)
                {
                    std::cout << "Failed at position (" << yxi << ", output feature = " << ofi << ", batch = " << bi << "): "
                        << output_vals[i] /*get_value(expected_ptr, i)*/ << " != " << output_ptr[i] << std::endl;
                    return;
                }
            }
        }
    }

#undef USE_OLD_WEIGHTS_FORMAT
}

using TestParamType_convolution_gpu = ::testing::tuple<int,   // 0 - Filter size
                                                       int,   // 1 - Input features
                                                       int,   // 2 - Stride
                                                       int,   // 3 - Output padding
                                                       bool>; // 4 - With bias


using TestParamType_convolution_depthwise_gpu = ::testing::tuple<int,   // 0 - Input XY size
        int,   // 1 - Kernel sizeY
        int,   // 2 - Kernel sizeX
        int,   // 3 - Groups number
        int,   // 4 - Stride
        int,   // 5 - Output padding
        bool>; // 6 - With bias

struct convolution_gpu : public ::testing::TestWithParam<TestParamType_convolution_gpu>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_gpu> param_info)
    {
        // construct a readable name
        return std::to_string(testing::get<0>(param_info.param))
            + 'x' + std::to_string(testing::get<0>(param_info.param))
            + "_f" + std::to_string(testing::get<1>(param_info.param))
            + "_stride" + std::to_string(testing::get<2>(param_info.param))
            + "_pad" + std::to_string(testing::get<3>(param_info.param))
            + (testing::get<4>(param_info.param) ? "_bias" : "");
    }
};

struct convolution_gpu_fs_byx_fsv32 : public convolution_gpu {};

struct convolution_depthwise_gpu : public ::testing::TestWithParam<TestParamType_convolution_depthwise_gpu>
{
    static std::string
    PrintToStringParamName(testing::TestParamInfo<TestParamType_convolution_depthwise_gpu> param_info)
    {
        // construct a readable name
        return "in" + std::to_string(testing::get<0>(param_info.param))
               + "x" + std::to_string(testing::get<0>(param_info.param))
               + "_k" + std::to_string(testing::get<1>(param_info.param))
               + 'x' + std::to_string(testing::get<2>(param_info.param))
               + "_f" + std::to_string(testing::get<3>(param_info.param))
               + "_stride" + std::to_string(testing::get<4>(param_info.param))
               + "_pad" + std::to_string(testing::get<5>(param_info.param))
               + (testing::get<6>(param_info.param) ? "_bias" : "");
    }
};

TEST_P(convolution_gpu, b_fs_yx_fsv4)
{
    const int in_B = 2;
    const int in_X = 56;
    const int in_Y = 56;
    const int _OuD = 32;
    const int W_B = _OuD;

    // Kernel sizes
    int W_X = testing::get<0>(GetParam());
    int W_Y = W_X;

    // Convoluiton offset
    int offSet = -(W_X / 2);

    // Features
    int in_F = testing::get<1>(GetParam());
    int W_F = in_F;

    // Stride
    int stride = testing::get<2>(GetParam());

    // Output padding
    int output_padding = testing::get<3>(GetParam());

    // Biases
    bool with_bias = testing::get<4>(GetParam());

    engine engine;

    // Input data init
    std::vector<char> Data(in_B * in_F * in_X * in_Y);
    std::iota(Data.begin(), Data.end(), 0);
    auto input = memory::allocate(engine, {data_types::i8, format::bfyx, {in_B, in_F, in_X, in_Y}});
    set_values(input, std::move(Data));

    // Create a topology
    topology topology_ref(input_layout("input", input.get_layout()));
    topology topology_imad(input_layout("input", input.get_layout()));

    // Reorder
    topology_imad.add(reorder("reorder_in",
                              "input",
                              layout(data_types::i8, format::b_fs_yx_fsv4, {in_B, in_F, in_X, in_Y})));

    // Weights init
    std::vector<char> Weights(W_B * W_F * W_X * W_Y);
    std::iota(Weights.begin(), Weights.end(), 0);
    auto weights_gold =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    auto weights_imad =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    set_values(weights_gold, Weights);
    set_values(weights_imad, std::move(Weights));
    topology_ref.add(data("weights_gold", weights_gold));
    topology_imad.add(data("weights_imad", weights_imad));

    if (with_bias)
    {
        // Bias, Callibraiton, Quantization
        std::vector<float> vB(_OuD), vC(_OuD), vQ(_OuD);
        float x = 0.1f;
        std::generate(vB.begin(), vB.end(), [x]() mutable {
            x += 0.01f;
            if (x >= 0.9f)
                x = 0.1f;
            return x;
        });
        x = 0.2f;
        std::generate(vC.begin(), vC.end(), [x]() mutable {
            x += 0.01f;
            if (x >= 0.9f)
                x = 0.2f;
            return x;
        });
        x = 0.3f;
        std::generate(vQ.begin(), vQ.end(), [x]() mutable {
            x += 0.01f;
            if (x >= 0.9f)
                x = 0.3f;
            return x;
        });
        auto bias_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, _OuD, 1, 1}});
        auto bias_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, _OuD, 1, 1}});
        auto callib_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
        auto callib_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
        auto quant_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
        auto quant_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
        set_values(bias_gold, vB);
        set_values(bias_imad, std::move(vB));
        set_values(callib_gold, vC);
        set_values(callib_imad, std::move(vC));
        set_values(quant_gold, vQ);
        set_values(quant_imad, std::move(vQ));
        topology_ref.add(data("bias_gold", bias_gold),
                         data("callib_gold", callib_gold),
                         data("quant_gold", quant_gold));
        topology_imad.add(data("bias_imad", bias_imad),
                         data("callib_imad", callib_imad),
                         data("quant_imad", quant_imad));

        // Convolutions
        convolution conv_gold("conv_gold",
                              "input",
                              {"weights_gold"},
                              {"bias_gold"},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        convolution conv_imad("conv_imad",
                              "reorder_in",
                              {"weights_imad"},
                              {"bias_imad"},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        conv_gold.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        conv_imad.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        topology_ref.add(conv_gold);
        topology_imad.add(conv_imad);
    }
    else
    {
        // Convolutions
        convolution conv_gold(
            "conv_gold", "input", {"weights_gold"}, {1, 1, stride, stride}, {0, 0, offSet, offSet});
        convolution conv_imad(
            "conv_imad", "reorder_in", {"weights_imad"}, {1, 1, stride, stride}, {0, 0, offSet, offSet});
        conv_gold.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        conv_imad.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        topology_ref.add(conv_gold);
        topology_imad.add(conv_imad);
    }

    // Reorder
    topology_imad.add(reorder("reorder_out",
                              "conv_imad",
                              layout(data_types::i8,
                                     format::bfyx,
                                     {in_B, W_B, (in_X + stride - 1) / stride, (in_Y + stride - 1) / stride},
                                     padding({0, 0, output_padding, output_padding}, 0.0f))));

    // Network build
    build_options build_opt_ref, build_opt_imad;
    build_opt_ref.set_option(build_option::optimize_data(false));
    build_opt_imad.set_option(build_option::optimize_data(true));

    network network_ref(engine, topology_ref, build_opt_ref);
    network network_imad(engine, topology_imad, build_opt_imad);

    network_ref.set_input_data("input", input);
    auto outputs_ref = network_ref.execute();
    network_imad.set_input_data("input", input);
    auto outputs_imad = network_imad.execute();

    auto out_gold = outputs_ref.find("conv_gold");
    auto out_test = outputs_imad.find("reorder_out");

    ASSERT_NE(out_gold, outputs_ref.end());
    ASSERT_NE(out_test, outputs_imad.end());

    auto gold_ptr = out_gold->second.get_memory().pointer<char>();
    auto test_ptr = out_test->second.get_memory().pointer<char>();

    ASSERT_EQ(gold_ptr.size(), test_ptr.size());
    for (size_t i = 0; i < gold_ptr.size(); i++)
    {
        ASSERT_EQ(gold_ptr[i], test_ptr[i]);
    }
}
TEST_P(convolution_gpu, b_fs_yx_fsv4_i8_to_u8)
{
    const int in_B = 2;
    const int in_X = 56;
    const int in_Y = 56;
    const int _OuD = 32;
    const int W_B = _OuD;

    // Kernel sizes
    int W_X = testing::get<0>(GetParam());
    int W_Y = W_X;

    // Convoluiton offset
    int offSet = -(W_X / 2);

    // Features
    int in_F = testing::get<1>(GetParam());
    int W_F = in_F;

    // Stride
    int stride = testing::get<2>(GetParam());

    // Output padding
    int output_padding = testing::get<3>(GetParam());

    // Biases
    bool with_bias = testing::get<4>(GetParam());

    engine engine;

    // Input data init
    std::vector<char> Data(in_B * in_F * in_X * in_Y);
    std::iota(Data.begin(), Data.end(), 0);
    auto input = memory::allocate(engine, {data_types::i8, format::bfyx, {in_B, in_F, in_X, in_Y}});
    set_values(input, std::move(Data));

    // Create a topology
    topology topology_ref(input_layout("input", input.get_layout()));
    topology topology_imad(input_layout("input", input.get_layout()));

    // Reorder
    topology_imad.add(reorder("reorder_in",
                         "input",
                         layout(data_types::i8, format::b_fs_yx_fsv4, {in_B, in_F, in_X, in_Y})));

    // Weights init
    std::vector<char> Weights(W_B * W_F * W_X * W_Y);
    std::iota(Weights.begin(), Weights.end(), 0);
    auto weights_gold =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    auto weights_imad =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    set_values(weights_gold, Weights);
    set_values(weights_imad, std::move(Weights));
    topology_ref.add(data("weights_gold", weights_gold));
    topology_imad.add(data("weights_imad", weights_imad));

    // Bias, Callibraiton, Quantization

    std::vector<float> vQ(_OuD);
    float x = 0.3f;
    std::generate(vQ.begin(), vQ.end(), [x]() mutable {
        x += 0.01f;
        if (x >= 0.9f)
            x = 0.3f;
        return x;
    });
    auto bias_gold = memory::allocate(engine, {data_types::i32, format::bfyx, {1, _OuD, 1, 1}});
    auto bias_imad = memory::allocate(engine, {data_types::i32, format::bfyx, {1, _OuD, 1, 1}});
    auto callib_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
    auto callib_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
    auto quant_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
    auto quant_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});

    set_values(quant_gold, vQ);
    set_values(quant_imad, std::move(vQ));
    topology_ref.add(data("quant_gold", quant_gold));
    topology_imad.add(data("quant_imad", quant_imad));


    if (with_bias)
    {
        std::vector<int> vB(_OuD);

        int i = 1;
        std::generate(vB.begin(), vB.end(), [i]() mutable {
            i += 1;
            if (i >= 128)
                i = 1;
            return i;
        });
        set_values(bias_gold, vB);
        set_values(bias_imad, std::move(vB));
        topology_ref.add(data("bias_gold",bias_gold));
        topology_imad.add(data("bias_imad",bias_imad));
        // Convolutions
        convolution conv_gold("conv_gold",
                              "input",
                              {"weights_gold"},
                              {"bias_gold"},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        convolution conv_imad("conv_imad",
                              "reorder_in",
                              {"weights_imad"},
                              {"bias_imad"},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        conv_gold.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        conv_imad.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);

        conv_gold.output_data_type = data_types::u8;
        conv_imad.output_data_type = data_types::u8;

        topology_ref.add(conv_gold);
        topology_imad.add(conv_imad);
    }
    else
    {
        // Convolutions
        convolution conv_gold("conv_gold",
                              "input",
                              {"weights_gold"},
                              {},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        convolution conv_imad("conv_imad",
                              "reorder_in",
                              {"weights_imad"},
                              {},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});

        conv_gold.output_data_type = data_types::u8;
        conv_imad.output_data_type = data_types::u8;

        conv_gold.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        conv_imad.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        topology_ref.add(conv_gold);
        topology_imad.add(conv_imad);
    }

    // Reorder
    topology_imad.add(reorder("reorder_out",
                              "conv_imad",
                              layout(data_types::u8,
                                     format::bfyx,
                                     {in_B, W_B, (in_X + stride - 1) / stride, (in_Y + stride - 1) / stride},
                                     padding({0, 0, output_padding, output_padding}, 0.0f))));

    // Network build
    build_options build_opt_ref, build_opt_imad;
    build_opt_ref.set_option(build_option::optimize_data(false));
    build_opt_imad.set_option(build_option::optimize_data(true));

    network network_ref(engine, topology_ref, build_opt_ref);
    network network_imad(engine, topology_imad, build_opt_imad);

    network_ref.set_input_data("input", input);
    auto outputs_ref = network_ref.execute();
    network_imad.set_input_data("input", input);
    auto outputs_imad = network_imad.execute();

    auto out_gold = outputs_ref.find("conv_gold");
    auto out_test = outputs_imad.find("reorder_out");

    ASSERT_NE(out_gold, outputs_ref.end());
    ASSERT_NE(out_test, outputs_imad.end());

    auto gold_ptr = out_gold->second.get_memory().pointer<char>();
    auto test_ptr = out_test->second.get_memory().pointer<char>();

    ASSERT_EQ(gold_ptr.size(), test_ptr.size());
    for (size_t i = 0; i < gold_ptr.size(); i++)
    {
        ASSERT_EQ(gold_ptr[i], test_ptr[i]);
    }
}
TEST_P(convolution_gpu, b_fs_yx_fsv4_i8_to_fp32)
{
    const int in_B = 2;
    const int in_X = 56;
    const int in_Y = 56;
    const int _OuD = 32;
    const int W_B = _OuD;

    // Kernel sizes
    int W_X = testing::get<0>(GetParam());
    int W_Y = W_X;

    // Convoluiton offset
    int offSet = -(W_X / 2);

    // Features
    int in_F = testing::get<1>(GetParam());
    int W_F = in_F;

    // Stride
    int stride = testing::get<2>(GetParam());

    // Output padding
    int output_padding = testing::get<3>(GetParam());

    // Biases
    bool with_bias = testing::get<4>(GetParam());

    engine engine;

    // Input data init
    std::vector<char> Data(in_B * in_F * in_X * in_Y);
    std::iota(Data.begin(), Data.end(), 0);
    auto input = memory::allocate(engine, {data_types::i8, format::bfyx, {in_B, in_F, in_X, in_Y}});
    set_values(input, std::move(Data));

    // Create a topology
    topology topology_ref(input_layout("input", input.get_layout()));
    topology topology_imad(input_layout("input", input.get_layout()));

    // Reorder
    topology_imad.add(reorder("reorder_in",
                         "input",
                         layout(data_types::i8, format::b_fs_yx_fsv4, {in_B, in_F, in_X, in_Y})));

    // Weights init
    std::vector<char> Weights(W_B * W_F * W_X * W_Y);
    std::iota(Weights.begin(), Weights.end(), 0);
    auto weights_gold =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    auto weights_imad =
        memory::allocate(engine, {data_types::i8, format::bfyx, {W_B, W_F, W_X, W_Y}});
    set_values(weights_gold, Weights);
    set_values(weights_imad, std::move(Weights));
    topology_ref.add(data("weights_gold", weights_gold));
    topology_imad.add(data("weights_imad", weights_imad));

    // Bias, Callibraiton, Quantization

    std::vector<float> vQ(_OuD);
    float x = 0.3f;
    std::generate(vQ.begin(), vQ.end(), [x]() mutable {
        x += 0.01f;
        if (x >= 0.9f)
            x = 0.3f;
        return x;
    });
    auto bias_gold = memory::allocate(engine, {data_types::i32, format::bfyx, {1, _OuD, 1, 1}});
    auto bias_imad = memory::allocate(engine, {data_types::i32, format::bfyx, {1, _OuD, 1, 1}});
    auto callib_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
    auto callib_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
    auto quant_gold = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});
    auto quant_imad = memory::allocate(engine, {data_types::f32, format::bfyx, {1, 1, _OuD, 1}});

    set_values(quant_gold, vQ);
    set_values(quant_imad, std::move(vQ));
    topology_ref.add(data("quant_gold", quant_gold));
    topology_imad.add(data("quant_imad", quant_imad));


    if (with_bias)
    {
        std::vector<int> vB(_OuD);

        int i = 1;
        std::generate(vB.begin(), vB.end(), [i]() mutable {
            i += 1;
            if (i >= 128)
                i = 1;
            return i;
        });
        set_values(bias_gold, vB);
        set_values(bias_imad, std::move(vB));
        topology_ref.add(data("bias_gold",bias_gold));
        topology_imad.add(data("bias_imad",bias_imad));
        // Convolutions
        convolution conv_gold("conv_gold",
                              "input",
                              {"weights_gold"},
                              {"bias_gold"},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        convolution conv_imad("conv_imad",
                              "reorder_in",
                              {"weights_imad"},
                              {"bias_imad"},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        conv_gold.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        conv_imad.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);

        conv_gold.output_data_type = data_types::f32;
        conv_imad.output_data_type = data_types::f32;

        topology_ref.add(conv_gold);
        topology_imad.add(conv_imad);
    }
    else
    {
        // Convolutions
        convolution conv_gold("conv_gold",
                              "input",
                              {"weights_gold"},
                              {},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});
        convolution conv_imad("conv_imad",
                              "reorder_in",
                              {"weights_imad"},
                              {},
                              {1, 1, stride, stride},
                              {0, 0, offSet, offSet});

        conv_gold.output_data_type = data_types::f32;
        conv_imad.output_data_type = data_types::f32;

        conv_gold.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        conv_imad.output_padding = padding({0, 0, output_padding, output_padding}, 0.0f);
        topology_ref.add(conv_gold);
        topology_imad.add(conv_imad);
    }

    // Reorder
    topology_imad.add(reorder("reorder_out",
                              "conv_imad",
                              layout(data_types::f32,
                                     format::bfyx,
                                     {in_B, W_B, (in_X + stride - 1) / stride, (in_Y + stride - 1) / stride},
                                     padding({0, 0, output_padding, output_padding}, 0.0f))));

    // Network build
    build_options build_opt_ref, build_opt_imad;
    build_opt_ref.set_option(build_option::optimize_data(false));
    build_opt_imad.set_option(build_option::optimize_data(true));

    network network_ref(engine, topology_ref, build_opt_ref);
    network network_imad(engine, topology_imad, build_opt_imad);

    network_ref.set_input_data("input", input);
    auto outputs_ref = network_ref.execute();
    network_imad.set_input_data("input", input);
    auto outputs_imad = network_imad.execute();

    auto out_gold = outputs_ref.find("conv_gold");
    auto out_test = outputs_imad.find("reorder_out");

    ASSERT_NE(out_gold, outputs_ref.end());
    ASSERT_NE(out_test, outputs_imad.end());

    auto gold_ptr = out_gold->second.get_memory().pointer<float>();
    auto test_ptr = out_test->second.get_memory().pointer<float>();

    ASSERT_EQ(gold_ptr.size(), test_ptr.size());
    for (size_t i = 0; i < gold_ptr.size(); i++)
    {
        ASSERT_EQ(gold_ptr[i], test_ptr[i]);
    }
}

// Select particular test cases
//INSTANTIATE_TEST_CASE_P(convolution_gpu_imad,
//                        convolution_gpu,
//                        ::testing::Values(
//                            // Filter size, Input features, Stride, Output padding, With bias
//                            TestParamType_convolution_gpu(1, 32, 1, 0, false),
//                            TestParamType_convolution_gpu(3, 32, 1, 0, false),
//                            TestParamType_convolution_gpu(7,  3, 1, 0, false),
//                            TestParamType_convolution_gpu(1, 32, 1, 0, true),
//                            TestParamType_convolution_gpu(3, 32, 1, 0, true),
//                            TestParamType_convolution_gpu(7,  3, 1, 0, true),
//                            TestParamType_convolution_gpu(1, 32, 1, 1, false),
//                            TestParamType_convolution_gpu(3, 32, 1, 1, false),
//                            TestParamType_convolution_gpu(7,  3, 1, 1, false),
//                            TestParamType_convolution_gpu(1, 32, 2, 0, false),
//                            TestParamType_convolution_gpu(3, 32, 2, 0, false),
//                            TestParamType_convolution_gpu(7,  3, 2, 0, false),
//                            TestParamType_convolution_gpu(3, 64, 2, 1, true)),
//                        convolution_gpu::PrintToStringParamName);
//// or test all combinations
//INSTANTIATE_TEST_CASE_P(convolution_gpu_imad,
//                        convolution_gpu,
//                        ::testing::Combine(::testing::Values(1, 3, 7),    // Filter size
//                                           ::testing::Values(3, 32),      // Input features
//                                           ::testing::Values(1, 2),       // Stride
//                                           ::testing::Values(0, 1),       // Output padding
//                                           ::testing::Values(false, true) // With bias
//                                           ),
//                        convolution_gpu::PrintToStringParamName);

INSTANTIATE_TEST_CASE_P(convolution_gpu_test,
                        convolution_gpu_fs_byx_fsv32,
                        ::testing::Values(
                                // Filter size, Input features, Stride, Output padding, With bias
                                TestParamType_convolution_gpu(1, 20, 1, 0, false),
                                TestParamType_convolution_gpu(3, 80, 1, 0, false),
                                TestParamType_convolution_gpu(1, 32, 1, 0, true),
                                TestParamType_convolution_gpu(3, 32, 1, 0, true),
                                TestParamType_convolution_gpu(1, 32, 1, 1, false),
                                TestParamType_convolution_gpu(3, 32, 1, 1, false),
                                TestParamType_convolution_gpu(1, 32, 2, 0, false),
                                TestParamType_convolution_gpu(3, 32, 2, 0, false),
                                TestParamType_convolution_gpu(3, 64, 2, 1, true)),
                        convolution_gpu::PrintToStringParamName);

TEST_P(convolution_gpu_fs_byx_fsv32, fs_byx_fsv32)
{
    const auto& engine = get_test_engine();

    if (!engine.get_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    const int batch_num = 2;
    const int input_xy = 5;
    const int input_f = testing::get<1>(GetParam());
    const int output_f = 64;
    const int filter_xy = testing::get<0>(GetParam());
    const int stride = testing::get<2>(GetParam());
    const int output_padding = testing::get<3>(GetParam());
    const bool with_bias = testing::get<4>(GetParam());
    const int input_offset = -(filter_xy / 2);

    const int output_xy = 1 + (input_xy + 2 * (-input_offset) - filter_xy) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = generate_random_4d<FLOAT16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = memory::allocate(engine, { data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, input_f, filter_xy, filter_xy);
    auto weights_data = generate_random_4d<FLOAT16>(output_f, input_f, filter_xy, filter_xy, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = memory::allocate(engine, { data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<FLOAT16>(batch_num, VVVF<FLOAT16>(output_f));

    topology topology(
        input_layout("input", input_mem.get_layout()),
        data("weights_fsv", weights_mem));

    // Reorder input to fs_byx_fsv32
    topology.add(reorder("input_fsv", "input", { data_types::f16, format::fs_b_yx_fsv32, input_size }));

    if (with_bias)
    {
        // Generate bias data
        auto biases_size = tensor(1, output_f, 1, 1);
        auto biases_data = generate_random_1d<FLOAT16>(output_f, -1, 1);
        auto biases_mem = memory::allocate(engine, { data_types::f16, format::bfyx, biases_size });
        set_values(biases_mem, biases_data);

        // Calculate reference values with bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],
                    stride, stride, biases_data[ofi],
                    1, 1,                               // dilation
                    -input_offset, -input_offset,       // input padding
                    output_padding, output_padding);
            }
        }

        topology.add(data("biases_fsv", biases_mem));

        auto conv_fsv = convolution("conv_fsv", "input_fsv", { "weights_fsv" }, { "biases_fsv" },
                                    { 1, 1, stride, stride }, { 0, 0, input_offset, input_offset });
        conv_fsv.output_padding = padding({ 0, 0, output_padding, output_padding }, 0.f);

        topology.add(conv_fsv);
    }
    else
    {
        // Calculate reference values without bias
        for (auto bi = 0; bi < batch_num; ++bi)
        {
            for (auto ofi = 0; ofi < output_f; ++ofi)
            {
                reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],
                    stride, stride,
                    0,                                  // bias
                    1, 1,                               // dilation
                    -input_offset, -input_offset,       // input padding
                    output_padding, output_padding);
            }
        }

        auto conv_fsv = convolution("conv_fsv", "input_fsv", { "weights_fsv" },
            { 1, 1, stride, stride }, { 0, 0, input_offset, input_offset });
        conv_fsv.output_padding = padding({ 0, 0, output_padding, output_padding }, 0.f);

        topology.add(conv_fsv);
    }


    build_options options;
    implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
    options.set_option(build_option::force_implementations({ {"conv_fsv", conv_impl} }));
    options.set_option(build_option::optimize_data(true));
    network network(engine, topology, options);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_fsv").get_memory();
    auto out_ptr = out_mem.pointer<FLOAT16>();

    ASSERT_EQ(out_mem.get_layout().format, format::fs_b_yx_fsv32);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_xy; ++yi)
                for (int xi = 0; xi < output_xy; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];
                    auto val = out_ptr[(fi / 32) * batch_num * output_xy * output_xy * 32 +
                                        bi * output_xy * output_xy * 32 +
                                        yi * output_xy * 32 +
                                        xi * 32 +
                                        fi % 32];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    EXPECT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", xi = " << xi << ", yi = " << yi << std::endl;
                    }
                }
}

INSTANTIATE_TEST_CASE_P(convolution_depthwise_gpu,
                        convolution_depthwise_gpu,
                        ::testing::Values(
                                // Input size, Filter size Y, Filter size X, groups, Stride, Output padding, With bias
                                // Stride testing
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 0, false),
                                // Different Features testing
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 20, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 25, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 33, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 35, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 45, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 65, 1, 0, false),
                                // Different filter's sizes testing
                                TestParamType_convolution_depthwise_gpu(5, 3, 2, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 1, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 2, 3, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 1, 3, 16, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 2, 16, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 1, 16, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 2, 3, 16, 2, 0, false),
                                TestParamType_convolution_depthwise_gpu(5, 1, 3, 16, 2, 0, false),
                                // Input FeatureMap testing
                                TestParamType_convolution_depthwise_gpu(20, 3, 3, 50, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(30, 3, 3, 50, 1, 0, false),
                                TestParamType_convolution_depthwise_gpu(55, 3, 3, 50, 1, 0, false),
                                // Output padding testing + strides
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 1, 1, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 2, 2, false),
                                TestParamType_convolution_depthwise_gpu(5, 3, 3, 32, 3, 3, false)
                                ),
                        convolution_depthwise_gpu::PrintToStringParamName);

TEST_P(convolution_depthwise_gpu, depthwise_conv_fs_b_yx_fsv32)
{
    const auto& engine = get_test_engine();

    if (!engine.get_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        EXPECT_EQ(1, 1);
        return;
    }

    const int batch_num = 2;
    const int input_xy = testing::get<0>(GetParam());
    const int groups = testing::get<3>(GetParam());
    const int input_f = groups;
    const int output_f = groups;
    const int filter_y = testing::get<1>(GetParam());
    const int filter_x = testing::get<2>(GetParam());
    const int stride = testing::get<4>(GetParam());
    const int output_padding = testing::get<5>(GetParam());
    const int input_offset_y = -(filter_y / 2);
    const int input_offset_x = -(filter_x / 2);

    const int output_y = 1 + (input_xy + 2 * (-input_offset_y) - filter_y) / stride + 2 * output_padding;
    const int output_x = 1 + (input_xy + 2 * (-input_offset_x) - filter_x) / stride + 2 * output_padding;

    auto input_size = tensor(batch_num, input_f, input_xy, input_xy);
    auto input_data = generate_random_4d<FLOAT16>(batch_num, input_f, input_xy, input_xy, -1, 1);
    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto input_mem = memory::allocate(engine, { data_types::f16, format::bfyx, input_size });
    set_values(input_mem, input_data_bfyx);

    auto weights_size = tensor(output_f, 1, filter_x, filter_y);
    auto weights_data = generate_random_4d<FLOAT16>(output_f, 1, filter_y, filter_x, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    auto weights_mem = memory::allocate(engine, { data_types::f16, format::bfyx, weights_size });
    set_values(weights_mem, weights_data_bfyx);

    // Will be used to store reference values calculated in branches depending on bias
    auto reference_result = VVVVF<FLOAT16>(batch_num, VVVF<FLOAT16>(output_f));

    topology topology(
            input_layout("input", input_mem.get_layout()),
            data("weights_fsv", weights_mem));

    // Reorder input to fs_byx_fsv32
    topology.add(reorder("input_fsv", "input", { data_types::f16, format::fs_b_yx_fsv32, input_size }));

    // Calculate reference values without bias
    for (auto bi = 0; bi < batch_num; ++bi)
    {
        for (auto ofi = 0; ofi < output_f; ++ofi)
        {
            reference_result[bi][ofi] = reference_convolve(
                    input_data[bi], weights_data[ofi],  // input, weights
                    stride, stride,                     // strides
                    0,                                  // bias
                    1, 1,                               // dilation
                    -input_offset_y, -input_offset_x,   // input padding
                    output_padding, output_padding,     // output_padding
                    ofi, ofi + 1,                       // f_begin, f_end
                    true);                              // depthwise
        }
    }

    auto conv_fsv = convolution("conv_fsv", "input_fsv", { "weights_fsv" }, groups,
                                { 1, 1, stride, stride }, { 0, 0, input_offset_x, input_offset_y });
    conv_fsv.output_padding = padding({ 0, 0, output_padding, output_padding }, 0.f);

    topology.add(conv_fsv);

    build_options options;
    options.set_option(build_option::optimize_data(true));
    implementation_desc conv_impl = { format::fs_b_yx_fsv32, "" };
    options.set_option(build_option::force_implementations({ {"conv_fsv", conv_impl} }));
    network network(engine, topology, options);

    network.set_input_data("input", input_mem);

    network.execute();

    auto out_mem = network.get_output("conv_fsv").get_memory();
    auto out_ptr = out_mem.pointer<FLOAT16>();

    ASSERT_EQ(out_mem.get_layout().format, format::fs_b_yx_fsv32);

    for (int bi = 0; bi < batch_num; ++bi)
        for (int fi = 0; fi < output_f; ++fi)
            for (int yi = 0; yi < output_y; ++yi)
                for (int xi = 0; xi < output_x; ++xi)
                {
                    auto val_ref = reference_result[bi][fi][yi][xi];
                    auto val = out_ptr[(fi / 32) * batch_num * output_y * output_x * 32 +
                                       bi * output_y * output_x * 32 +
                                       yi * output_x * 32 +
                                       xi * 32 +
                                       fi % 32];
                    auto equal = are_equal(val_ref, val, 1e-2f);
                    EXPECT_TRUE(equal);
                    if (!equal)
                    {
                        std::cout << "At b = " << bi << ", fi = " << fi << ", yi = " << yi << ", xi = " << xi << std::endl;
                    }
                }
}

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_test_base : public testing::Test {
public:
    virtual topology build_topology(const cldnn::engine& engine) {
        auto input_lay = layout(input_type(), input_format(), input_size());
        auto wei_lay = layout(weights_type(), format::bfyx, weights_size());

        auto wei_mem = memory::allocate(engine, wei_lay);
        auto weights_flat = flatten_4d(format::bfyx, _weights);
        set_values(wei_mem, weights_flat);

        auto topo = topology();
        topo.add(input_layout("input", input_lay));
        topo.add(data("weights", wei_mem));
        if (!has_bias()) {
            auto conv_prim = convolution(
                "conv",
                "input",
                { "weights" },
                tensor(batch(0), feature(0), spatial(_stride_x, _stride_y)),
                tensor(batch(0), feature(0), spatial(_offset_x, _offset_y)),
                tensor(batch(0), feature(0), spatial(_dilation_x, _dilation_y)));
            conv_prim.output_data_type = output_type();
            topo.add(conv_prim);
        } else {
            auto bias_lay = layout(output_type(), format::bfyx, tensor(feature(output_features())));
            auto bias_mem = memory::allocate(engine, bias_lay);
            set_values(bias_mem, _bias);
            topo.add(data("bias", bias_mem));
            auto conv_prim = convolution(
                "conv",
                "input",
                { "weights" },
                { "bias" },
                tensor(batch(0), feature(0), spatial(_stride_x, _stride_y)),
                tensor(batch(0), feature(0), spatial(_offset_x, _offset_y)),
                tensor(batch(0), feature(0), spatial(_dilation_x, _dilation_y)));
            conv_prim.output_data_type = output_type();
            topo.add(conv_prim);
        }

        return topo;
    }

    virtual primitive_id output_primitive_id() const {
        return "conv";
    }

    void run_expect(const VVVVF<OutputT>& expected) {
        auto engine = get_test_engine();

        auto topo = build_topology(engine);

        auto build_opts = build_options(
            build_option::optimize_data(true)
        );
        auto prog = program(engine, topo, build_opts);

        auto net = network(prog, 0);

        auto input_lay = layout(input_type(), input_format(), input_size());
        auto input_mem = memory::allocate(engine, input_lay);
        std::vector<InputT> input_flat(input_lay.get_linear_size(), static_cast<InputT>(0));
        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < input_features(); ++fi)
                for (size_t yi = 0; yi < input_y(); ++yi)
                    for (size_t xi = 0; xi < input_x(); ++xi) {
                        tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        size_t offset = input_lay.get_linear_offset(coords);
                        input_flat[offset] = _input[bi][fi][yi][xi];
                    }
        set_values(input_mem, input_flat);

        net.set_input_data("input", input_mem);
        auto result = net.execute();
        auto out_mem = result.at(output_primitive_id()).get_memory();
        auto out_lay = out_mem.get_layout();
        auto out_ptr = out_mem.cldnn::memory::template pointer<OutputT>();

        ASSERT_EQ(out_lay.data_type, output_type());
        ASSERT_EQ(out_lay.size.batch[0], expected.size());
        ASSERT_EQ(out_lay.size.feature[0], expected[0].size());
        ASSERT_EQ(out_lay.size.spatial[1], expected[0][0].size());
        ASSERT_EQ(out_lay.size.spatial[0], expected[0][0][0].size());

        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < output_features(); ++fi)
                for (size_t yi = 0; yi < expected[0][0].size(); ++yi)
                    for (size_t xi = 0; xi < expected[0][0][0].size(); ++xi) {
                        tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        size_t offset = out_lay.get_linear_offset(coords);

                        EXPECT_EQ(out_ptr[offset], expected[bi][fi][yi][xi])
                            << "at b= " << bi << ", f= " << fi << ", y= " << yi << ", x= " << xi;
                    }
    }

    void set_input(format::type fmt, VVVVF<InputT> input) {
        _input_fmt = fmt;
        _input = std::move(input);
    }

    void set_weights(VVVVF<WeightsT> weights) {
        _weights = std::move(weights);
    }

    void set_bias(VF<OutputT> bias) {
        _bias = std::move(bias);
    }

    void set_strides(int stride_x, int stride_y) {
        _stride_x = stride_x;
        _stride_y = stride_y;
    }

    void set_offsets(int offset_x, int offset_y) {
        _offset_x = offset_x;
        _offset_y = offset_y;
    }

    void set_dilation(int dilation_x, int dilation_y) {
        _dilation_x = dilation_x;
        _dilation_y = dilation_y;
    }

protected:
    VVVVF<InputT> _input;
    VVVVF<WeightsT> _weights;
    VF<OutputT> _bias;
    format::type _input_fmt;
    int _stride_x, _stride_y;
    int _offset_x, _offset_y;
    int _dilation_x, _dilation_y;

    size_t batch_num() const { return _input.size(); }
    size_t input_features() const { return _input[0].size(); }
    size_t input_x() const { return _input[0][0][0].size(); }
    size_t input_y() const { return _input[0][0].size(); }
    size_t output_features() const { return _weights.size(); }
    size_t filter_x() const { return _weights[0][0][0].size(); }
    size_t filter_y() const { return _weights[0][0].size(); }

    bool has_bias() { return _bias.size() > 0; }

    data_types input_type() const { return type_to_data_type<InputT>::value; }
    format input_format() const { return _input_fmt; }
    tensor input_size() const {
        return tensor(TensorValue(batch_num()),
                      TensorValue(input_features()),
                      TensorValue(input_x()),
                      TensorValue(input_y()));
    }

    data_types weights_type() const { return type_to_data_type<WeightsT>::value; }
    tensor weights_size() const {
        return tensor(TensorValue(output_features()),
                      TensorValue(input_features()),
                      TensorValue(filter_x()),
                      TensorValue(filter_y()));
    }

    data_types output_type() const { return type_to_data_type<OutputT>::value; }
};

using convolution_random_test_params = std::tuple<
    size_t,                     // batch
    size_t,                     // input features
    size_t,                     // output features
    std::tuple<size_t, size_t>, // input x, y
    std::tuple<size_t, size_t>, // filter x, y
    std::tuple<int, int>,       // stride x, y
    std::tuple<int, int>,       // offset x, y
    std::tuple<int, int>,       // dilation x, y
    bool,                       // with bias
    format::type                // input format
>;

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_random_test_base : public convolution_test_base<InputT, WeightsT, OutputT> {
public:
    virtual VVVVF<OutputT> calculate_reference() {
        VVVVF<OutputT> expected = VVVVF<OutputT>(this->batch_num(), VVVF<OutputT>(this->output_features()));
        for (size_t bi = 0; bi < this->batch_num(); ++bi)
        for (size_t fi = 0; fi < this->output_features(); ++fi) {
            auto bias = this->has_bias() ? this->_bias[fi] : static_cast<OutputT>(0);
            expected[bi][fi] = reference_convolve<InputT, OutputT, WeightsT>(
                this->_input[bi],
                this->_weights[fi],
                this->_stride_y,
                this->_stride_x,
                static_cast<float>(bias),
                this->_dilation_y,
                this->_dilation_x,
                this->_offset_y,
                this->_offset_x);
        }
        return expected;
    }

    virtual void param_set_up(const convolution_random_test_params& params) {
        size_t b, in_f, out_f, in_x, in_y, f_x, f_y;
        int s_x, s_y, o_x, o_y, d_x, d_y;
        format::type in_format;
        bool w_bias;

        std::forward_as_tuple(
            b,
            in_f,
            out_f,
            std::forward_as_tuple(in_x, in_y),
            std::forward_as_tuple(f_x, f_y),
            std::forward_as_tuple(s_x, s_y),
            std::forward_as_tuple(o_x, o_y),
            std::forward_as_tuple(d_x, d_y),
            w_bias,
            in_format) = params;

        auto input_data = generate_random_4d<InputT>(b, in_f, in_y, in_x, -256, 256);
        auto weights_data = generate_random_4d<WeightsT>(out_f, in_f, f_y, f_x, -256, 256);
        auto bias_data = w_bias ? generate_random_1d<OutputT>(out_f, -256, 256) : VF<OutputT>();

        this->set_input(in_format, std::move(input_data));
        this->set_weights(std::move(weights_data));
        this->set_bias(std::move(bias_data));
        this->set_strides(s_x, s_y);
        this->set_offsets(o_x, o_y);
        this->set_dilation(d_x, d_y);
    }

    void run_random(const convolution_random_test_params& params) {
        param_set_up(params);

        VVVVF<OutputT> expected = calculate_reference();
        ASSERT_NO_FATAL_FAILURE(this->run_expect(expected));
    }

    // construct a readable name in format as follows:
    // <out format>_i<input>_w<weights>_s<stride>_ofs<offset>_d<dilation>_<bias>
    static std::string PrintToStringParamName(testing::TestParamInfo<convolution_random_test_params> param_info) {
        int Batch = (int)testing::get<0>(param_info.param);
        int iF = (int)testing::get<1>(param_info.param);
        int oF = (int)testing::get<2>(param_info.param);
        auto iSize = testing::get<3>(param_info.param);
        auto fSize = testing::get<4>(param_info.param);
        auto Stride = testing::get<5>(param_info.param);
        auto Offset = testing::get<6>(param_info.param);
        auto Dilation = testing::get<7>(param_info.param);
        bool Bias = testing::get<8>(param_info.param);
        format::type iType = testing::get<9>(param_info.param);  // input format

        return fmt_to_str(iType) + "_i" + std::to_string(Batch) + 'x' + std::to_string(iF) + 'x' +
               std::to_string(std::get<0>(iSize)) + 'x' + std::to_string(std::get<1>(iSize)) + "_w" +
               std::to_string(oF) + 'x' + std::to_string(iF) + 'x' + std::to_string(std::get<0>(fSize)) + 'x' +
               std::to_string(std::get<1>(fSize)) + "_s" + std::to_string(std::get<0>(Stride)) + 'x' +
               std::to_string(std::get<1>(Stride)) + "_ofs" + std::to_string(std::get<0>(Offset)) + 'x' +
               std::to_string(std::get<1>(Offset)) + "_d" + std::to_string(std::get<0>(Dilation)) + 'x' +
               std::to_string(std::get<1>(Dilation)) + (Bias ? "_bias" : "");
    }

};

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_random_test : public convolution_random_test_base<InputT, WeightsT, OutputT>
                              , public testing::WithParamInterface<convolution_random_test_params> {};


using convolution_random_test_s8s8f32 = convolution_random_test<int8_t, int8_t, float>;
using convolution_random_test_u8s8f32 = convolution_random_test<uint8_t, int8_t, float>;

TEST_P(convolution_random_test_s8s8f32, random) {
    ASSERT_NO_FATAL_FAILURE(run_random(GetParam()));
}

TEST_P(convolution_random_test_u8s8f32, random) {
    ASSERT_NO_FATAL_FAILURE(run_random(GetParam()));
}

INSTANTIATE_TEST_CASE_P(
    b_fs_yx_fsv4,
    convolution_random_test_s8s8f32,
    testing::Combine(
        testing::Values(1, 2),                                                             // batch
        testing::Values(3, 32),                                                            // input features
        testing::Values(16, 32),                                                           // output features
        testing::Values(std::pair<size_t, size_t>(7, 7), std::pair<size_t, size_t>(8, 8)), // input x, y
        testing::Values(std::pair<size_t, size_t>(1, 1), std::pair<size_t, size_t>(3, 3)), // filter x, y
        testing::Values(std::pair<int, int>(1, 1), std::pair<int, int>(2, 2)),             // strides x, y
        testing::Values(std::pair<int, int>(0, 0)),                                        // offsets x, y
        testing::Values(std::pair<int, int>(1, 1)),                                        // dilation x, y
        testing::Values(false, true),                                                      // bias
        testing::Values(format::b_fs_yx_fsv4)                                              // input format
    ),
    convolution_random_test_s8s8f32::PrintToStringParamName);

INSTANTIATE_TEST_CASE_P(
    b_fs_yx_fsv4,
    convolution_random_test_u8s8f32,
    testing::Combine(
        testing::Values(1, 2),                                                             // batch
        testing::Values(3, 32),                                                            // input features
        testing::Values(16, 32),                                                           // output features
        testing::Values(std::pair<size_t, size_t>(7, 7), std::pair<size_t, size_t>(8, 8)), // input x, y
        testing::Values(std::pair<size_t, size_t>(1, 1), std::pair<size_t, size_t>(3, 3)), // filter x, y
        testing::Values(std::pair<int, int>(1, 1), std::pair<int, int>(2, 2)),             // strides x, y
        testing::Values(std::pair<int, int>(0, 0)),                                        // offsets x, y
        testing::Values(std::pair<int, int>(1, 1)),                                        // dilation x, y
        testing::Values(false, true),                                                      // bias
        testing::Values(format::b_fs_yx_fsv4)                                              // input format
    ),
    convolution_random_test_u8s8f32::PrintToStringParamName);

INSTANTIATE_TEST_CASE_P(
    b_fs_yx_fsv4_1x1_lwg_opt,
    convolution_random_test_s8s8f32,
    testing::Combine(
        testing::Values(1),                               // batch
        testing::Values(128, 256, 512),                   // input features
        testing::Values(64),                              // output features
        testing::Values(std::pair<size_t, size_t>(3, 3)), // input x, y
        testing::Values(std::pair<size_t, size_t>(1, 1)), // filter x, y
        testing::Values(std::pair<int, int>(1, 1)),       // strides x, y
        testing::Values(std::pair<int, int>(0, 0)),       // offsets x, y
        testing::Values(std::pair<int, int>(1, 1)),       // dilation x, y
        testing::Values(false),                           // bias
        testing::Values(format::b_fs_yx_fsv4)             // input format
    ),
    convolution_random_test_s8s8f32::PrintToStringParamName);

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_scale_random_test : public convolution_random_test<InputT, WeightsT, OutputT> {
public:
    using parent = convolution_random_test<InputT, WeightsT, OutputT>;

    virtual primitive_id output_primitive_id() const {
        return "scale_wa_reorder";
    }

    topology build_topology(const cldnn::engine& engine) override {
        topology topo = parent::build_topology(engine);

        auto scale_lay = layout(this->output_type(), format::bfyx, tensor(batch(1), feature(this->output_features())));
        auto shift_lay = layout(this->output_type(), format::bfyx, tensor(batch(1), feature(this->output_features())));

        auto scale_mem = memory::allocate(engine, scale_lay);
        auto shift_mem = memory::allocate(engine, shift_lay);

        set_values(scale_mem, _scale);
        set_values(shift_mem, _shift);

        topo.add(cldnn::data("scale_scale", scale_mem));
        topo.add(cldnn::data("scale_shift", shift_mem));
        topo.add(cldnn::scale("scale", "conv", "scale_scale", "scale_shift"));
        // Work-around since if scale is output it will not be fused
        topo.add(cldnn::reorder("scale_wa_reorder", "scale", format::bfyx, this->output_type()));
        return topo;
    }

    VVVVF<OutputT> calculate_reference() override {
        auto expected = parent::calculate_reference();

        for (size_t bi = 0; bi < this->batch_num(); ++bi)
        for (size_t fi = 0; fi < this->output_features(); ++fi) {
            expected[bi][fi] = reference_scale_post_op<OutputT>(expected[bi][fi], _scale[fi], _shift[fi]);
        }
        return expected;
    }

    void param_set_up(const convolution_random_test_params& params) override {
        parent::param_set_up(params);

        _scale = generate_random_1d<OutputT>(this->output_features(), -1, 1);
        _shift = generate_random_1d<OutputT>(this->output_features(), 128, 128);
    }
protected:
    VF<OutputT> _scale;
    VF<OutputT> _shift;
};

using convolution_scale_random_test_s8s8f32 = convolution_scale_random_test<int8_t, int8_t, float>;
using convolution_scale_random_test_u8s8f32 = convolution_scale_random_test<uint8_t, int8_t, float>;

TEST_P(convolution_scale_random_test_s8s8f32, random) {
    ASSERT_NO_FATAL_FAILURE(run_random(GetParam()));
}

TEST_P(convolution_scale_random_test_u8s8f32, random) {
    ASSERT_NO_FATAL_FAILURE(run_random(GetParam()));
}

INSTANTIATE_TEST_CASE_P(
    b_fs_yx_fsv4,
    convolution_scale_random_test_s8s8f32,
    testing::Combine(
        testing::Values(1, 2),                                                             // batch
        testing::Values(3, 32),                                                            // input features
        testing::Values(16, 32),                                                           // output features
        testing::Values(std::pair<size_t, size_t>(7, 7), std::pair<size_t, size_t>(8, 8)), // input x, y
        testing::Values(std::pair<size_t, size_t>(1, 1), std::pair<size_t, size_t>(3, 3)), // filter x, y
        testing::Values(std::pair<int, int>(1, 1), std::pair<int, int>(2, 2)),             // strides x, y
        testing::Values(std::pair<int, int>(0, 0)),                                        // offsets x, y
        testing::Values(std::pair<int, int>(1, 1)),                                        // dilation x, y
        testing::Values(false, true),                                                      // bias
        testing::Values(format::b_fs_yx_fsv4)                                              // input format
    ),
    convolution_scale_random_test_s8s8f32::PrintToStringParamName);

INSTANTIATE_TEST_CASE_P(
    b_fs_yx_fsv4,
    convolution_scale_random_test_u8s8f32,
    testing::Combine(
        testing::Values(1, 2),                                                             // batch
        testing::Values(3, 32),                                                            // input features
        testing::Values(16, 32),                                                           // output features
        testing::Values(std::pair<size_t, size_t>(7, 7), std::pair<size_t, size_t>(8, 8)), // input x, y
        testing::Values(std::pair<size_t, size_t>(1, 1), std::pair<size_t, size_t>(3, 3)), // filter x, y
        testing::Values(std::pair<int, int>(1, 1), std::pair<int, int>(2, 2)),             // strides x, y
        testing::Values(std::pair<int, int>(0, 0)),                                        // offsets x, y
        testing::Values(std::pair<int, int>(1, 1)),                                        // dilation x, y
        testing::Values(false, true),                                                      // bias
        testing::Values(format::b_fs_yx_fsv4)                                              // input format
    ),
    convolution_scale_random_test_u8s8f32::PrintToStringParamName);

template <typename InputT, typename WeightsT, typename OutputT>
class convolution_asymm_weights_data_random_test : public convolution_random_test<InputT, WeightsT, OutputT> {
    using parent = convolution_random_test<InputT, WeightsT, OutputT>;

    virtual primitive_id output_primitive_id() const {
       return "conv_wa_reorder";
    }

    topology build_topology(const cldnn::engine& engine) override {
        auto input_lay = layout(this->input_type(), this->input_format(), this->input_size());
        auto wei_lay = layout(this->weights_type(), format::bfyx, this->weights_size());
        auto data_zp_lay = layout(this->input_type(), format::bfyx, tensor(batch(1), feature(this->input_features()), spatial(1, 1)));
        auto wei_zp_lay = layout(this->weights_type(), format::bfyx, tensor(batch(this->output_features()), feature(1), spatial(1, 1)));

        auto wei_mem = memory::allocate(engine, wei_lay);
        auto data_zp_mem = memory::allocate(engine, data_zp_lay);
        auto wei_zp_mem = memory::allocate(engine, wei_zp_lay);
        auto weights_flat = flatten_4d(format::bfyx, this->_weights);
        set_values(wei_mem, weights_flat);
        set_values(data_zp_mem, _data_zp);
        set_values(wei_zp_mem, _weights_zp);

        auto topo = topology();
        topo.add(input_layout("input", input_lay));
        topo.add(data("weights", wei_mem));
        topo.add(data("data_zp", data_zp_mem));
        topo.add(data("weights_zp", wei_zp_mem));
        auto input_asymm_prim = eltwise("input_asymm", "input", "data_zp", eltwise_mode::sub);
        auto weights_asymm_prim = eltwise("weights_asymm", "weights", "weights_zp", eltwise_mode::sub);
        input_asymm_prim.output_data_type = data_types::f32;
        weights_asymm_prim.output_data_type = data_types::f32;
        topo.add(input_asymm_prim);
        topo.add(weights_asymm_prim);
        if (!this->has_bias()) {
            auto conv_prim = convolution(
                "conv",
                "input_asymm",
                { "weights_asymm" },
                tensor(batch(0), feature(0), spatial(this->_stride_x, this->_stride_y)),
                tensor(batch(0), feature(0), spatial(this->_offset_x, this->_offset_y)),
                tensor(batch(0), feature(0), spatial(this->_dilation_x, this->_dilation_y)));
            conv_prim.output_data_type = this->output_type();
            topo.add(conv_prim);
        } else {
            auto bias_lay = layout(this->output_type(), format::bfyx, tensor(feature(this->output_features())));
            auto bias_mem = memory::allocate(engine, bias_lay);
            set_values(bias_mem, this->_bias);
            topo.add(data("bias", bias_mem));
            auto conv_prim = convolution(
                "conv",
                "input_asymm",
                { "weights_asymm" },
                { "bias" },
                tensor(batch(0), feature(0), spatial(this->_stride_x, this->_stride_y)),
                tensor(batch(0), feature(0), spatial(this->_offset_x, this->_offset_y)),
                tensor(batch(0), feature(0), spatial(this->_dilation_x, this->_dilation_y)));
            conv_prim.output_data_type = this->output_type();
            topo.add(conv_prim);
        }
        topo.add(reorder("conv_wa_reorder", "conv", format::bfyx, this->output_type()));

        return topo;
    }

    VVVVF<OutputT> calculate_reference() override {
        VVVVF<OutputT> expected = VVVVF<OutputT>(this->batch_num(), VVVF<OutputT>(this->output_features()));
        for (size_t bi = 0; bi < this->batch_num(); ++bi)
            for (size_t fi = 0; fi < this->output_features(); ++fi) {
                auto bias = this->has_bias() ? this->_bias[fi] : static_cast<OutputT>(0);
                expected[bi][fi] = reference_convolve<InputT, OutputT, WeightsT>(
                    this->_input[bi],
                    this->_weights[fi],
                    this->_stride_y,
                    this->_stride_x,
                    static_cast<float>(bias),
                    this->_dilation_y,
                    this->_dilation_x,
                    this->_offset_y,
                    this->_offset_x,
                    0,
                    0,
                    0,
                    0,
                    false,
                    _data_zp,
                    _weights_zp[fi]);
            }
        return expected;
    }

    void param_set_up(const convolution_random_test_params& params) override {
        parent::param_set_up(params);

        _data_zp = generate_random_1d<InputT>(this->input_features(), -128, 128);
        _weights_zp = generate_random_1d<WeightsT>(this->output_features(), -128, 128);
    }

protected:
    VF<InputT> _data_zp;
    VF<WeightsT> _weights_zp;
};

using convolution_asymm_random_test_s8s8f32 = convolution_asymm_weights_data_random_test<int8_t, int8_t, float>;
using convolution_asymm_random_test_u8s8f32 = convolution_asymm_weights_data_random_test<uint8_t, int8_t, float>;

TEST_P(convolution_asymm_random_test_s8s8f32, random) {
    ASSERT_NO_FATAL_FAILURE(run_random(GetParam()));
}

TEST_P(convolution_asymm_random_test_u8s8f32, random) {
    ASSERT_NO_FATAL_FAILURE(run_random(GetParam()));
}

INSTANTIATE_TEST_CASE_P(
    basic_asymm,
    convolution_asymm_random_test_s8s8f32,
    testing::Combine(
        testing::Values(1, 2),                                                             // batch
        testing::Values(3, 32),                                                            // input features
        testing::Values(16, 32),                                                           // output features
        testing::Values(std::pair<size_t, size_t>(7, 7), std::pair<size_t, size_t>(8, 8)), // input x, y
        testing::Values(std::pair<size_t, size_t>(1, 1), std::pair<size_t, size_t>(3, 3)), // filter x, y
        testing::Values(std::pair<int, int>(1, 1), std::pair<int, int>(2, 2)),             // strides x, y
        testing::Values(std::pair<int, int>(0, 0)),                                        // offsets x, y
        testing::Values(std::pair<int, int>(1, 1)),                                        // dilation x, y
        testing::Values(false, true),                                                      // bias
        testing::Values(format::bfyx, format::b_fs_yx_fsv32)                               // input format
    ),
    convolution_asymm_random_test_s8s8f32::PrintToStringParamName);

INSTANTIATE_TEST_CASE_P(
    basic_asymm,
    convolution_asymm_random_test_u8s8f32,
    testing::Combine(
        testing::Values(1, 2),                                                             // batch
        testing::Values(3, 32),                                                            // input features
        testing::Values(16, 32),                                                           // output features
        testing::Values(std::pair<size_t, size_t>(7, 7), std::pair<size_t, size_t>(8, 8)), // input x, y
        testing::Values(std::pair<size_t, size_t>(1, 1), std::pair<size_t, size_t>(3, 3)), // filter x, y
        testing::Values(std::pair<int, int>(1, 1), std::pair<int, int>(2, 2)),             // strides x, y
        testing::Values(std::pair<int, int>(0, 0)),                                        // offsets x, y
        testing::Values(std::pair<int, int>(1, 1)),                                        // dilation x, y
        testing::Values(false, true),                                                      // bias
        testing::Values(format::bfyx, format::b_fs_yx_fsv32)                               // input format
    ),
    convolution_asymm_random_test_u8s8f32::PrintToStringParamName);

class convolution_test : public tests::generic_test
{

public:

    static void TearDownTestCase()
    {
        for (auto generic_params : all_generic_params)
        {
            delete generic_params;
        }

        all_layer_params.clear();
        all_test_params.clear();
    }

    static std::vector<std::shared_ptr<cldnn::primitive>> generate_specific_test_params()
    {
        // TODO: check split

        // TODO: check convolution without bias

        const std::vector<primitive_id>& weights = { "input1" };
        const std::vector<primitive_id>& bias = { "input2" };

        std::vector<tensor> stride_sizes = { tensor(1, 1, 1, 1), tensor(1, 1, 2, 3), tensor(1, 1, 4, 1), tensor(1, 1, 5, 5) };
        std::vector<tensor> dilation_sizes = { tensor(1, 1, 1, 1), tensor(1, 1, 5, 4), tensor(1, 1, 1, 3), tensor(1, 1, 7, 2) };
        std::vector<tensor> input_offset_sizes = { tensor(0, 0, 0, 0), tensor(0, 0, 2, 2), tensor(0, 0, -5, -2), tensor(0, 0, 3, -3) };

        // No padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "input0", weights, bias, stride_sizes[0], input_offset_sizes[0], dilation_sizes[0]));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "input0", weights, bias, stride_sizes[1], input_offset_sizes[1], dilation_sizes[1]));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "input0", weights, bias, stride_sizes[2], input_offset_sizes[2], dilation_sizes[2]));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "input0", weights, bias, stride_sizes[3], input_offset_sizes[3], dilation_sizes[3]));

        // Input padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "reorder0", weights, bias, stride_sizes[1], input_offset_sizes[1], dilation_sizes[1]));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "reorder0", weights, bias, stride_sizes[3], input_offset_sizes[3], dilation_sizes[3]));

        // Output padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "input0", weights, bias, stride_sizes[1], input_offset_sizes[1], dilation_sizes[1], { { 0, 0, 2, 4 },{ 0, 0, 0, 19 } }));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "input0", weights, bias, stride_sizes[2], input_offset_sizes[2], dilation_sizes[2], { { 0, 0, 1, 0 },{ 0, 0, 13, 9 } }));

        // Input + Output padding
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "reorder0", weights, bias, stride_sizes[0], input_offset_sizes[0], dilation_sizes[0], { { 0, 0, 1, 5 },{ 0, 0, 19, 4 } }));
        all_layer_params.emplace_back(new convolution("convolution_no_relu", "reorder0", weights, bias, stride_sizes[3], input_offset_sizes[3], dilation_sizes[3], { { 0, 0, 1, 2 },{ 0, 0, 3, 4 } }));

        return all_layer_params;
    }

    static std::vector<std::tuple<tests::test_params*, std::shared_ptr<cldnn::primitive>>> generate_all_test_params()
    {
        generate_specific_test_params();

        std::vector<cldnn::format> input_formats = { cldnn::format::bfyx, cldnn::format::yxfb };
        std::vector<cldnn::format> weights_formats = { cldnn::format::bfyx, cldnn::format::yxfb };

        std::vector<int32_t> output_features_sizes = { 1, 3, 16 };
        std::vector<cldnn::tensor> kernel_sizes = { tensor(1, 1, 1, 1), tensor(1, 1, 4, 7), tensor(1, 1, 5, 3) };

        std::vector<tensor> input_tensor_size = { tensor(1, 5, 59, 72), tensor(8, 3, 63, 56), tensor(16, 2, 50, 50), tensor(32, 1, 44, 62) };

        auto data_types = test_data_types();

        for (cldnn::data_types data_type : data_types)
        {
            for (cldnn::format input_format : input_formats)
            {
                for (cldnn::format weights_format : weights_formats)
                {
                    cldnn::build_options network_build_options;
                    if (input_format == cldnn::format::bfyx)
                    {
                        network_build_options.set_option(cldnn::build_option::optimize_data(true));
                    }
                    for (cldnn::tensor input_size : input_tensor_size)
                    {
                        for (cldnn::tensor kernel_size : kernel_sizes)
                        {
                            for (auto output_features : output_features_sizes)
                            {
                                test_params* params = new test_params(data_type, input_format, input_size.batch[0], input_size.feature[0], tensor(1, 1, input_size.spatial[0], input_size.spatial[1]), network_build_options);
                                int input_features = params->input_layouts[0].size.feature[0];
                                params->input_layouts.push_back(cldnn::layout(params->data_type, weights_format, cldnn::tensor(output_features, input_features, kernel_size.spatial[0], kernel_size.spatial[1]))); // weights
                                params->input_layouts.push_back(cldnn::layout(params->data_type, params->fmt, cldnn::tensor(1, 1, output_features, 1))); // biases
                                all_generic_params.push_back(params);
                            }
                        }
                    }
                }
            }
        }

        // Create all the combinations for the test.
        for (const auto& layer_param : all_layer_params)
        {
            for (tests::test_params* test_param : all_generic_params)
            {
                all_test_params.push_back(std::make_tuple(test_param, layer_param));
            }
        }

        return all_test_params;
    }

    virtual bool is_format_supported(cldnn::format format)
    {
        return ((format == cldnn::format::bfyx) || (format == cldnn::format::yxfb));
    }

    virtual cldnn::tensor get_expected_output_tensor()
    {
        auto convolution = std::static_pointer_cast<const cldnn::convolution>(layer_params);
        tensor input_size = generic_params->input_layouts[0].size;
        tensor dilation = convolution->dilation;
        tensor stride = convolution->stride;
        tensor input_offset = convolution->input_offset;
        tensor weights_size = generic_params->input_layouts[1].size;

        int kernel_extent_y = dilation.spatial[1] * (weights_size.spatial[1] - 1) + 1;
        int kernel_extent_x = dilation.spatial[0] * (weights_size.spatial[0] - 1) + 1;

        // Calculate output size
        int output_size_y = 1 + (input_size.spatial[1] - kernel_extent_y - 2 * input_offset.spatial[1]) / stride.spatial[1];
        int output_size_x = 1 + (input_size.spatial[0] - kernel_extent_x - 2 * input_offset.spatial[0]) / stride.spatial[0];
        int output_features = weights_size.batch[0];

        return cldnn::tensor(input_size.batch[0], output_features, output_size_x, output_size_y);
    }

    virtual void prepare_input_for_test(std::vector<cldnn::memory>& inputs)
    {
        if (generic_params->data_type == data_types::f32)
        {
            prepare_input_for_test_typed<float>(inputs);
        }
        else
        {
            prepare_input_for_test_typed<FLOAT16>(inputs);
        }
    }

    template<typename Type>
    void prepare_input_for_test_typed(std::vector<cldnn::memory>& inputs)
    {
        int k = (generic_params->data_type == data_types::f32) ? 8 : 4;

        // Update inputs.
        auto input = inputs[0];
        auto input_size = inputs[0].get_layout().size;
        VVVVF<Type> input_rnd = generate_random_4d<Type>(input_size.batch[0], input_size.feature[0], input_size.spatial[1], input_size.spatial[0], -2, 2, k);
        VF<Type> input_rnd_vec = flatten_4d<Type>(input.get_layout().format, input_rnd);
        set_values(input, input_rnd_vec);

        // Update weights.
        auto weight_input = inputs[1];
        auto weight_size = inputs[1].get_layout().size;
        VVVVF<Type> weight_rnd = generate_random_4d<Type>(weight_size.batch[0], weight_size.feature[0], weight_size.spatial[1], weight_size.spatial[0], -2, 2, k);
        VF<Type> weight_rnd_vec = flatten_4d<Type>(weight_input.get_layout().format, weight_rnd);
        set_values(weight_input, weight_rnd_vec);

        // Update biases.
        auto bias_input = inputs[2];
        auto bias_size = inputs[2].get_layout().size;
        VF<Type> bias_rnd = generate_random_1d<Type>(bias_size.spatial[0], -2, 2, k);
        set_values(bias_input, bias_rnd);
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<cldnn::memory>& inputs)
    {
        // Output reference is always bfyx.

        auto convolution = std::static_pointer_cast<const cldnn::convolution>(layer_params);

        data_types dt = inputs[0].get_layout().data_type;

        tensor input_size = inputs[0].get_layout().size;
        tensor dilation = convolution->dilation;
        tensor stride = convolution->stride;
        tensor input_offset = convolution->input_offset;
        tensor weights_size = inputs[1].get_layout().size;
        padding output_padding = convolution->output_padding;

        tensor output_size = get_expected_output_tensor();

        // Calculate output size
        int output_size_y = output_size.spatial[1];
        int output_size_x = output_size.spatial[0];
        int output_features = weights_size.batch[0];
        int input_features = weights_size.feature[0];

        auto output = memory::allocate( engine, cldnn::layout(dt, cldnn::format::bfyx, output_size, output_padding) );

        auto input_mem = inputs[0].pointer<Type>();
        auto weights_mem = inputs[1].pointer<Type>();
        auto bias_mem = inputs[2].pointer<Type>();
        auto output_mem = output.pointer<Type>();

        tensor output_buffer_size = output.get_layout().get_buffer_size();

        // Initialized output with zeros.
        std::fill(output_mem.begin(), output_mem.end(), static_cast<Type>(0));

        // Add the bias
        for (int b = 0; b < input_size.batch[0]; b++)
        {
            for (int out_f = 0; out_f < output_features; out_f++)
            {
                for (int y = 0; y < output_size_y; y++)
                {
                    for (int x = 0; x < output_size_x; x++)
                    {
                        int output_index = (b * output_buffer_size.feature[0] + out_f) * output_buffer_size.spatial[1] * output_buffer_size.spatial[0];
                        tensor lower_output_padding = convolution->output_padding.lower_size();
                        output_index += (lower_output_padding.spatial[1] + y) * output_buffer_size.spatial[0] + lower_output_padding.spatial[0] + x;

                        output_mem[output_index] += bias_mem[out_f];
                    }
                }
            }
        }

        const auto input0_desc = get_linear_memory_desc(inputs[0].get_layout());
        const auto input1_desc = get_linear_memory_desc(inputs[1].get_layout());

        // Convolve with weights
        for (int b = 0; b < input_size.batch[0]; b++)
        {
            int input_bi = b;
            for (int out_f = 0; out_f < output_features; out_f++)
            {
                for (int in_f = 0; in_f < input_features; in_f++)
                {
                    int input_fi = in_f;
                    for (int y = 0; y < output_size_y; y++)
                    {
                        for (int x = 0; x < output_size_x; x++)
                        {
                            int output_bi = b;
                            int output_fi = out_f;
                            int output_yi = y;
                            int output_xi = x;
                            int output_index = (output_bi * output_buffer_size.feature[0] + output_fi) * output_buffer_size.spatial[1] * output_buffer_size.spatial[0];
                            tensor lower_output_padding = convolution->output_padding.lower_size();
                            output_index += (lower_output_padding.spatial[1] + output_yi) * output_buffer_size.spatial[0] + lower_output_padding.spatial[0] + output_xi;

                            for (int kernel_y = 0; kernel_y < weights_size.spatial[1]; kernel_y++)
                            {
                                int input_yi = y * stride.spatial[1] + input_offset.spatial[1] + kernel_y * dilation.spatial[1];
                                if ((input_yi < 0) || (input_yi >= input_size.spatial[1]))
                                {
                                    continue;
                                }

                                for (int kernel_x = 0; kernel_x < weights_size.spatial[0]; kernel_x++)
                                {
                                    int input_xi = x * stride.spatial[0] + input_offset.spatial[0] + kernel_x * dilation.spatial[0];
                                    if ((input_xi < 0) || (input_xi >= input_size.spatial[0]))
                                    {
                                        continue;
                                    }

                                    size_t input_index = get_linear_index(inputs[0].get_layout(), input_bi, input_fi, input_yi, input_xi, input0_desc);

                                    int weight_bi = out_f;
                                    int weight_fi = in_f;
                                    int weight_yi = kernel_y;
                                    int weight_xi = kernel_x;
                                    size_t weight_index = get_linear_index(inputs[1].get_layout(), weight_bi, weight_fi, weight_yi, weight_xi, input1_desc);
                                    output_mem[output_index] += input_mem[input_index] * weights_mem[weight_index];
                                }
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

    virtual memory generate_reference(const std::vector<cldnn::memory>& inputs)
    {
        if (generic_params->data_type == data_types::f32)
        {
            return generate_reference_typed<float>(inputs);
        }
        else
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;
    static std::vector<std::tuple<tests::test_params*, std::shared_ptr<cldnn::primitive>>> all_test_params;
};

std::vector<tests::test_params*> convolution_test::all_generic_params = {};
std::vector<std::shared_ptr<cldnn::primitive>> convolution_test::all_layer_params = {};
std::vector<std::tuple<tests::test_params*, std::shared_ptr<cldnn::primitive>>> convolution_test::all_test_params = {};

TEST_P(convolution_test, CONVOLUTION)
{
    run_single_test();
}

INSTANTIATE_TEST_CASE_P(DISABLED_CONVOLUTION,
                        convolution_test,
                        ::testing::ValuesIn(convolution_test::generate_all_test_params()),
                        tests::generic_test::custom_param_name_functor());
