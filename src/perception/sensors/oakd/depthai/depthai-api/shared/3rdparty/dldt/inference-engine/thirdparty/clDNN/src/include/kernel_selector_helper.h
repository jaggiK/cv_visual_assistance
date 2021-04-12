// Copyright (c) 2016-2018 Intel Corporation
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

#pragma once

#include "api/cldnn.hpp"
#include "api/tensor.hpp"
#include "api/eltwise.hpp"
#include "api/scale.hpp"
#include "api/quantize.hpp"
#include "api/activation.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include "tensor_type.h"
#include "error_handler.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

using namespace cldnn;

namespace cldnn {
enum class data_types : size_t;
enum class tuning_mode;
struct format;
struct layout;
struct program_impl;
struct program_node;
}  // namespace cldnn

namespace kernel_selector {
using n_dims = kernel_selector::Tensor::NDims;
using kernel_data = kernel_selector::KernelData;
using kernel_string = kernel_selector::KernelString;
using cl_kernel_data = kernel_selector::clKernelData;
using kernel_arguments = kernel_selector::Arguments;
using kernel_argument_element = kernel_selector::ArgumentDescriptor;
using kernel_argument_types = kernel_selector::ArgumentDescriptor::Types;
using kernel_scalar_arguments = kernel_selector::Scalars;
using kernel_scalar_argument_types = kernel_selector::ScalarDescriptor::Types;

using data_type = kernel_selector::Datatype;
using kernel_type = kernel_selector::KernelType;
using weights_type = kernel_selector::WeightsType;
using activation_function = kernel_selector::ActivationFunction;
using pool_type = kernel_selector::PoolType;
using pool_remainder = kernel_selector::PoolRemainder;
using argm_axis = kernel_selector::ArgMaxMinAxis;
using argm_output = kernel_selector::ArgMaxMinOut;
using argm_sort = kernel_selector::ArgMaxMinSortType;
using lookt_axis = kernel_selector::LookUpTableAxis;
using lrn_mode = kernel_selector::LRNMode;
using normalize_mode = kernel_selector::NormalizeMode;
using mvn_mode = kernel_selector::MVNMode;
using kernel_divider_mode = kernel_selector::KernelDividerMode;
using eltwise_mode = kernel_selector::EltwiseMode;
using eltwise_input_mode = kernel_selector::EltwiseInputMode;
using softmax_dim = kernel_selector::SoftmaxDim;
using mean_subtruct_mode = kernel_selector::MeanSubtractMode;
using mean_op = kernel_selector::MeanOp;
using concat_axis = kernel_selector::ConcatAxis;
using tile_axis = kernel_selector::TileAxis;
using tuning_mode = kernel_selector::TuningMode;
using sample_type = kernel_selector::ResampleType;
using border_type = kernel_selector::BorderType;
using gather_axis = kernel_selector::GatherAxis;
using reduce_mode = kernel_selector::ReduceMode;

using data_tensor = kernel_selector::DataTensor;
using weights_tensor = kernel_selector::WeightsTensor;
template <typename T>
using dim_tensor = kernel_selector::DimTensor<T>;
using data_layout = kernel_selector::DataLayout;
using weights_layout = kernel_selector::WeightsLayout;
using multi_data_tensor = kernel_selector::MultiDataTensor;

using params = kernel_selector::Params;
using weights_reorder_params = kernel_selector::WeightsReorderParams;
using generic_kernel_params = kernel_selector::GenericKernelParams;

struct training_params;
}  // namespace kernel_selector

kernel_selector::data_type to_data_type(data_types dt);
data_types from_data_type(kernel_selector::data_type dt);
kernel_selector::weights_type to_weights_type(data_types dt);
data_types from_weights_type(kernel_selector::weights_type dt);
kernel_selector::data_layout to_data_layout(format f);
cldnn::format from_data_layout(kernel_selector::data_layout l);
kernel_selector::weights_layout to_weights_layout(format f);
cldnn::format::type from_weights_layout(kernel_selector::weights_layout l);
kernel_selector::tuning_mode to_tuning_mode(cldnn::tuning_mode mode);
std::string to_host_version(const cldnn::version_t& version);
kernel_selector::data_tensor convert_data_tensor(const layout& l, uint32_t split = 1, const tensor view_offset = tensor {});
kernel_selector::weights_tensor convert_weights_tensor(const layout& l);
layout from_weights_tensor(const kernel_selector::weights_tensor& t);
kernel_selector::activation_function get_kernel_selector_activation_param(activation_func activation_func);
kernel_selector::activation_function get_kernel_selector_activation_grad_param(
    activation_grad_func activation_grad_func);

template <typename T = std::uint32_t>
kernel_selector::dim_tensor<T> convert_dim_vector(const tensor& t) {
    const auto& sizes = t.sizes(format::bfwzyx);
    return {static_cast<T>(sizes[0]),
            static_cast<T>(sizes[1]),
            static_cast<T>(sizes[2]),
            static_cast<T>(sizes[3]),
            static_cast<T>(sizes[4]),
            static_cast<T>(sizes[5])};
}

template <typename p_type>
inline void convert_activation_func_params(const p_type primitive, std::vector<kernel_selector::base_activation_params>& params) {
    const float negative_slope = primitive->activation_negative_slope;
    if (negative_slope != 0.0f) {
        params.emplace_back(kernel_selector::activation_function::RELU_NEGATIVE_SLOPE, negative_slope, 0.0f);
    } else {
        params.emplace_back(kernel_selector::activation_function::RELU, 0.0f, 0.0f);
    }
}

template <typename arg_t>
inline void convert_fused_activation_func_params(const arg_t& arg, std::vector<kernel_selector::base_activation_params>& params) {
    for (size_t i = 0; i < arg.get_fused_activations_funcs().size(); i++) {
        params.emplace_back(get_kernel_selector_activation_param(arg.get_fused_activations_funcs()[i]),
                            arg.get_fused_activations_params()[i].a,
                            arg.get_fused_activations_params()[i].b);
    }
}

template <typename p_type>
inline void convert_new_activation_func(const p_type primitive, std::vector<kernel_selector::base_activation_params>& params) {
    params.insert(params.begin(), {get_kernel_selector_activation_param(primitive->activation_function),
                                   primitive->additional_params.a,
                                   primitive->additional_params.b});
}

template <typename p_type>
inline void convert_new_activation_grad_func(const p_type primitive, std::vector<kernel_selector::base_activation_params>& params) {
    params.insert(params.begin(), {get_kernel_selector_activation_grad_param(primitive->activation_grad_function),
                                   primitive->additional_params.a,
                                   primitive->additional_params.b,
                                   true});
}

void set_params(const program_node& node, kernel_selector::params& params);

template <typename params_t, typename arg_t>
inline params_t get_default_params(const arg_t& arg, uint32_t split = 1) {
    params_t params;

    set_params(arg, params);

    const auto& input_layout = arg.input().get_output_layout();
    const auto& output_layout = arg.get_output_layout();

    params.inputs[0] = convert_data_tensor(input_layout, split);
    params.output = convert_data_tensor(output_layout, split);

    params.layerID = arg.id();

    convert_fused_activation_func_params(arg, params.activations);
    size_t op_id = 0;
    for (auto& fused_prim : arg.get_fused_primitives()) {
        kernel_selector::base_params::fused_operation_desc desc;
        desc.op_params = fused_prim.node->get_fuse_params();
        if (!desc.op_params) {
            CLDNN_ERROR_MESSAGE(arg.id(), "Invalid fused operation (" + fused_prim.node->id() + ") of type " +
                                           fused_prim.node->get_primitive()->type_string() );
        }
        desc.dep_idx_start = fused_prim.dep_start_idx;
        desc.dep_size = fused_prim.deps.size();
        desc.op_id = op_id++;
        desc.output_tensor = convert_data_tensor(fused_prim.output_layout);

        for (size_t i = desc.dep_idx_start; i < desc.dep_idx_start + desc.dep_size; i++) {
            desc.tensors.push_back(convert_data_tensor(arg.get_dependency(i).get_output_layout()));
        }

        params.fused_ops.push_back(desc);
    }

    return params;
}

template <typename params_t, typename arg_t>
inline params_t get_weights_bias_default_params(const arg_t& arg, uint32_t split = 1, uint32_t groups = 1) {
    params_t params = get_default_params<params_t>(arg, split);
    auto weights_layout = arg.weights().get_output_layout();
    if (groups != 1) {
        weights_layout.size.batch[0] /= static_cast<int>(groups);
    }
    params.weights = convert_weights_tensor(weights_layout);

    if (arg.bias_term()) {
        auto bias_layout = arg.bias().get_output_layout();
        // bias per output is not supported on cldnn
        if (groups != 1) {
            bias_layout.size.feature[0] /= static_cast<int>(groups);
        }
        params.bias.push_back(convert_data_tensor(bias_layout).FlattenFeatureAndSpatials());
    }

    return params;
}

void set_learning_params(const program_node& node, kernel_selector::training_params& params, bool use_momentum);

template <typename params_t, typename arg_t>
inline params_t get_default_learning_params(const arg_t& arg, uint32_t split = 1) {
    params_t params = get_weights_bias_default_params<params_t>(arg, split);
    set_learning_params(arg, params, arg.use_momentum());
    return params;
}

void set_optional_params(const program_impl& program, kernel_selector::optional_params& params);

template <typename optional_params_t>
inline optional_params_t get_default_optional_params(const program_impl& program) {
    optional_params_t params;
    set_optional_params(program, params);
    return params;
}

template <typename optional_params_t>
inline optional_params_t get_default_weights_bias_optional_params(const program_impl& program) {
    return get_default_optional_params<optional_params_t>(program);
}

template <typename optional_params_t>
inline optional_params_t get_default_learning_optional_params(const program_impl& program) {
    return get_default_weights_bias_optional_params<optional_params_t>(program);
}
