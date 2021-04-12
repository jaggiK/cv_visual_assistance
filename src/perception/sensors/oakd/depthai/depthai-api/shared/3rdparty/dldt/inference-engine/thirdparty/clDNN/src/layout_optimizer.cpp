/*
// Copyright (c) 2018-2019 Intel Corporation
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

#include "layout_optimizer.h"
#include "topology_impl.h"
#include "network_impl.h"
#include "primitive_inst.h"
#include "error_handler.h"

#include "data_inst.h"
#include "reorder_inst.h"
#include "generic_layer.hpp"
#include <sstream>

#include "eltwise_inst.h"
#include "pooling_inst.h"
#include "permute_inst.h"
#include <vector>
#include <memory>
#include <utility>

using namespace cldnn;

std::pair<std::shared_ptr<reorder>, bool> reorder_factory::get_reorder(primitive_id src_id,
                                                                       const layout& in_layout,
                                                                       const layout& out_layout
) {
    if (in_layout == out_layout)
        return std::make_pair(nullptr, true);

    cache_key ckey{ src_id, out_layout };
    auto itr = _cached_reorders.find(ckey);
    if (itr != _cached_reorders.end())
        return std::make_pair(itr->second, true);

    auto count = _cached_reorders.size();
    std::stringstream ss;
    ss << src_id << "_reorder_" << count;

    auto reorder = std::make_shared<cldnn::reorder>(ss.str(), src_id, out_layout);
    _cached_reorders[ckey] = reorder;

    return std::make_pair(reorder, false);
}

std::vector<std::pair<std::shared_ptr<primitive>, bool>> reorder_factory::get_weights_reorder(
    primitive_id input_id,
    const layout& old_layout,
    const kernel_selector::weights_reorder_params& reorder_params) {

    if (reorder_params.engine == kernel_selector::weights_reorder_params::Engine::NONE)
        return {};

    std::vector<std::pair<std::shared_ptr<primitive>, bool>> ret;

    if (reorder_params.engine == kernel_selector::weights_reorder_params::Engine::CPU &&
        reorder_params.cpuKernel != nullptr) {
        const auto intermediate_format = from_weights_layout(reorder_params.cpuKernel->GetExpectedInputLayout());
        const auto intermediate_type = from_weights_type(reorder_params.cpuKernel->GetExpectedInputType());
        if (intermediate_format != old_layout.format || intermediate_type != old_layout.data_type) {
            const layout intermediate_layout = { intermediate_type,
                                                intermediate_format,
                                                old_layout.size.transform(intermediate_format, 1) };

            auto reorder = get_reorder(input_id, old_layout, intermediate_layout);
            if (reorder.first) {
                ret.push_back(reorder);
                input_id = reorder.first->id;
            }
        }
    }

    // TODO: Add conversion of WeightsTensor to cldnn::tensor to have not flattened shape
    // layout expected_layout = from_weights_tensor(reorder_params.dest);

    auto new_dtype = from_weights_type(reorder_params.dest.GetDType());
    const auto bpp = data_type_traits::size_of(new_dtype);
    tensor expected_size = { 1, 1, 1, (tensor::value_type)(reorder_params.dest.PhysicalSizeInBytes() / bpp) };

    bool toImageType = IsImageType(reorder_params.dest.GetLayout());
    bool toDynamicLSTMType = IsDynamicLSTMType(reorder_params.dest.GetLayout());
    if (toImageType || toDynamicLSTMType)
        expected_size = old_layout.size;

    layout expected_layout = { new_dtype,
                              toImageType ? from_weights_layout(reorder_params.dest.GetLayout())
                                          : format::bfyx,  // simple linear format (flatten to x channel)
                              expected_size };

    cache_key ckey{ input_id, expected_layout };
    auto itr = _cached_generic_reorders.find(ckey);
    if (itr != _cached_generic_reorders.end()) {
        ret.push_back(std::make_pair(itr->second, true));
    } else {
        auto count = _cached_generic_reorders.size();
        std::stringstream ss;
        ss << input_id << "_generic_layer_" << count;

        auto reorder = std::make_shared<cldnn::generic_layer>(ss.str(), input_id, expected_layout, reorder_params);
        _cached_generic_reorders[ckey] = reorder;
        ret.push_back(std::make_pair(reorder, false));
    }

    return ret;
}

bool layout_optimizer::is_format_supported(program_node& node, format::type fmt) {
    if (node.is_type<fully_connected>() && fmt == format::byxf)
        return false;

    if (node.is_type<input_layout>())
        return node.get_output_layout().format == fmt;

    if (!_format_forcing.empty() && _format_forcing.count(node.id()))
        return _format_forcing.at(node.id()) == fmt;

    auto& engine = node.get_program().get_engine();
    auto prev_layout = node.get_output_layout();
    auto new_layout = prev_layout;
    new_layout.format = fmt;
    node.set_output_layout(new_layout, false);

    auto supported = node.type()->does_possible_implementation_exist(engine, node);

    node.set_output_layout(prev_layout, false);

    return supported;
}

bool layout_optimizer::can_fuse_reorder(program_node& prev, program_node& next, format fmt_prev, format fmt_next) {
    auto prev_simple = fmt_prev == format::bfyx || fmt_prev == format::byxf || fmt_prev == format::yxfb;
    auto next_simple = fmt_next == format::bfyx || fmt_next == format::byxf || fmt_next == format::yxfb;
    auto prev_output_layout = prev.get_output_layout();
    auto next_output_layout = next.get_output_layout();
    auto prev_dt = prev.get_output_layout().data_type;

    if (next.is_type<reorder>())
        return true;

    if (next.is_type<pooling>() &&
        ((prev_simple && next_simple) ||
        ((fmt_prev == format::b_fs_yx_fsv4 && fmt_next == format::bfyx) && (prev_dt == data_types::u8 || prev_dt == data_types::i8))))
        return true;

    if (next.is_type<eltwise>() && prev_simple && next_simple)
        return true;

    if (next.is_type<permute>() && (fmt_prev == format::bfzyx_f16 &&
        next_output_layout.size.batch[0] > 1 &&
        next_output_layout.size.feature[0] % 16 != 0)) {
        return true;
    }

    if (next.is_type<fully_connected>() &&
        (fmt_prev == format::bfyx || fmt_prev == format::yxfb ||
         fmt_prev == format::bfyx_f16 || fmt_prev == format::fs_b_yx_fsv32 ||
         fmt_prev == format::byxf_af32 || fmt_prev == format::b_fs_yx_fsv32 ||
         (fmt_prev == format::b_fs_yx_fsv4 &&
          prev_output_layout.size.feature[0] % 32 == 0 &&
          prev_output_layout.size.spatial[0] == 1 &&
          prev_output_layout.size.spatial[1] == 1)))
        return true;

    if (next.is_type<convolution>() && fmt_prev == format::byxf_af32 && fmt_next == format::b_fs_yx_fsv4 && next.as<convolution>().get_groups() != 1)
        return true;

    if (next.is_type<convolution>() && fmt_prev == format::byxf_af32 && fmt_next == format::bfyx)
        return true;

    if (next.is_type<convolution>() && fmt_prev == format::b_fs_yx_fsv4 && fmt_next == format::byxf_af32 && next.as<convolution>().get_groups() == 1)
        return true;

    if (next.is_type<quantize>() && fmt_prev == format::bfyx && prev.is_input() && prev_dt == data_types::u8)
        return true;

    if (next.is_type<convolution>() &&
        fmt_prev == format::bfyx &&
        ((fmt_next == format::fs_b_yx_fsv32 && next.as<convolution>().get_primitive()->groups == 1) ||
        (fmt_next == format::b_fs_yx_fsv32 && prev_output_layout.size.feature[0] == 3) ||
        (fmt_next == format::bfyx_f16 && next_output_layout.size.feature[0] >= 16 && prev_output_layout.size.feature[0] == 3)))
        return true;

    if (next.is_type<quantize>() && fmt_prev == format::bfyx && fmt_next == format::bfyx_f16)
        return true;

    if (next.is_type<convolution>() &&
        fmt_prev == format::bfyx && prev_output_layout.size.feature[0] == 3 &&
        (fmt_next == format::b_fs_yx_fsv4 || fmt_next == format::byxf_af32))
        return true;

    if (next.is_type<convolution>() &&
        fmt_prev == format::bfzyx &&
        ((fmt_next == format::bfzyx_f16 || fmt_next == format::bfzyx_b16f16) &&
            next_output_layout.size.feature[0] >= 16 && prev_output_layout.size.feature[0] == 3))
        return true;

    return false;
}

bool layout_optimizer::can_fuse_reorder_to_prev(program_node& prev, program_node& /*next*/, format /*fmt_prev*/, format fmt_next) {
    if (prev.is_type<reorder>())
        return true;

    if (prev.is_type<binary_convolution>() && fmt_next == format::bfyx_f16)
        return true;

    if (prev.is_type<quantize>() &&
        (fmt_next == format::b_fs_yx_fsv4 || fmt_next == format::b_fs_yx_fsv32 || fmt_next == format::b_fs_zyx_fsv32))
        return true;

    return false;
}


namespace {
bool should_use_winograd_2x3_s1(std::shared_ptr<const convolution> const& prim,
                                layout const& input_layout,
                                layout const& weights_layout,
                                bool output_size_handling_enabled) {
    // cases when NOT to use winograd
    if (input_layout.data_type != data_types::f16
        || input_layout.size.feature[0] % 64 != 0  // current algorithm is effective for ifm to be multiply of 64
        || weights_layout.size.spatial[0] != 3     // weights have to be 3x3 by definiton
        || weights_layout.size.spatial[1] != 3     // weights have to be 3x3 by definition
        || weights_layout.size.batch[0] % 64 != 0  // current algorithm is effective for ofm to be multiply of 64
        || prim->stride != tensor {1}               // stride has to be 1x1 by definition
        || prim->dilation != tensor {1}             // no support for dilation
        || prim->split() != 1                      // no support for splitted convolutions
        || (output_size_handling_enabled &&
            prim->with_output_size)                // no support for convolutions with user-specified output size
        || (input_layout.count() > 3000000)        // limit max input size as winograd consumes more memory
        || (input_layout.count() < 50000)          // limit min input size as winograd is not effective for small input
        || (input_layout.size.spatial[0] < 8 &&
            input_layout.size.spatial[1] < 8)      // disable winograd for small spatials as perf is poor
        || prim->groups != 1) {                    // disable winograd for groups
        return false;
    }
    return true;
}
}  // namespace

layout_optimizer::layout_optimizer(bool output_size_handling_enabled)
    : _optimization_attributes(), _output_size_handling_enabled(output_size_handling_enabled) {}

bool layout_optimizer::is_depthwise(const convolution_node& node) const {
        const int32_t output_channels = *node.get_output_layout().size.feature.begin();
        const int32_t input_channels = *node.get_dependency(0).get_output_layout().size.feature.begin();

        return (node.get_groups() == static_cast<uint32_t>(input_channels) && input_channels == output_channels && node.get_split() == 1)
                 || (node.get_split() == input_channels && node.get_groups() == 1);
}

bool layout_optimizer::convolution_bfyx_opt(layout const& output_layout,
                                            const layout& weights_layout,
                                            std::shared_ptr<const convolution> conv) {
    // A set of rules that define when bfyx mem format has better performance than yxfb
    if (output_layout.size.batch[0] == 16 || output_layout.size.batch[0] % 16 != 0 ||
        output_layout.data_type != data_types::f16 || weights_layout.size.batch[0] % 16 != 0 ||
        !((weights_layout.size.spatial[0] == 1 && weights_layout.size.spatial[1] == 1) ||
          (weights_layout.size.spatial[0] >= 5 && weights_layout.size.spatial[1] >= 5) ||
          (conv->stride.spatial[0] > 1 && conv->stride.spatial[1] > 1) ||
          (weights_layout.size.feature[0] <= 32 && output_layout.size.spatial[0] < 224 &&
           output_layout.size.spatial[1] < 224) ||
          (weights_layout.size.feature[0] <= 64 && output_layout.size.spatial[0] < 112 &&
           output_layout.size.spatial[1] < 112) ||
          (weights_layout.size.feature[0] <= 128 && output_layout.size.spatial[0] < 56 &&
           output_layout.size.spatial[1] < 56) ||
          (weights_layout.size.feature[0] <= 256 && output_layout.size.spatial[0] < 28 &&
           output_layout.size.spatial[1] < 28) ||
          (weights_layout.size.feature[0] <= 512 && output_layout.size.spatial[0] < 14 &&
           output_layout.size.spatial[1] < 14) ||
          (weights_layout.size.feature[0] <= 1024 && output_layout.size.spatial[0] <= 7 &&
           output_layout.size.spatial[1] <= 7)) ||
        // WA for AgeGender, which has one convolution that is better on yxfb, but due to additonal reorder overall
        // performance is worse than bfyx
        (output_layout.size.spatial[0] == 82 && output_layout.size.spatial[1] == 82) ||
        (_optimization_attributes.splitted_convolution && output_layout.size.batch[0] == 16) ||
        (!_optimization_attributes.splitted_convolution && output_layout.size.batch[0] >= 128) ||
        _optimization_attributes.bfyx_only_layer)
        return true;

    return false;
}

bool layout_optimizer::convolution_byxf_opt(const layout& input_layout,
                                            layout const& output_layout,
                                            const layout& weights_layout,
                                            const convolution_node& node) {
    auto conv = node.get_primitive();

    if (node.get_dependency(0).is_type<convolution>()) {
        auto& dep = node.get_dependency(0).as<convolution>();
        if (is_depthwise(dep))
            return false;
    }

    // A set of rules that define when byxf mem format has better performance
    if ((output_layout.data_type == data_types::f16 && weights_layout.size.spatial[0] == 1 &&
        conv->dilation == tensor { 1 } &&
        !node.get_transposed() &&
         node.get_groups() == 1 &&
         input_layout.size.feature[0] % 32 == 0 &&
         weights_layout.size.spatial[1] == 1 && output_layout.size.feature[0] % 64 == 0 &&
         weights_layout.size.batch[0] % 64 == 0 && conv->stride.spatial[0] == 1 && conv->stride.spatial[1] == 1 &&
         conv->input_offset.spatial[0] == 0 && conv->input_offset.spatial[1] == 0) ||
        // Winograd
        should_use_winograd_2x3_s1(conv, input_layout, weights_layout, _output_size_handling_enabled))
        return true;

    return false;
}

bool layout_optimizer::convolution_bfyx_f16_opt(layout const& input_layout,
                                                const layout& weights_layout,
                                                std::shared_ptr<const convolution> conv) {
    // A set of rules that define when bfyx_f16 mem format can be used
    if ((input_layout.data_type == data_types::f16 || input_layout.data_type == data_types::f32) &&
        weights_layout.data_type == input_layout.data_type &&
        input_layout.size.batch[0] == 1 &&
        input_layout.size.spatial[2] == 1 &&
        ((input_layout.size.feature[0] >= 16 && weights_layout.size.batch[0] >= 16) ||
         (input_layout.size.feature[0] == 3  && weights_layout.size.batch[0] >= 16)) &&
        ((conv->groups == 1 && conv->split() == 1) ||  // any conv with signle group
        ((conv->groups == static_cast<uint32_t>(input_layout.size.feature[0]) ||
          conv->split() == static_cast<int32_t>(input_layout.size.feature[0])) &&  // or depthwise 3x3 conv with stride 1 or 2
          weights_layout.size.spatial[0] == 3 &&
          weights_layout.size.spatial[1] == 3 &&
          (conv->stride.spatial[0] == 1 || conv->stride.spatial[0] == 2) &&
          (conv->stride.spatial[1] == 1 || conv->stride.spatial[1] == 2) &&
          (conv->stride.spatial[0] == conv->stride.spatial[1]) &&
          (conv->dilation == tensor{1}))))
        return true;
    return false;
}

bool layout_optimizer::convolution_bfzyx_f16_opt(layout const& input_layout,
    const layout& weights_layout,
    std::shared_ptr<const convolution> conv) {
    // A set of rules that define when bfzyx_f16 mem format can be used
    if ((input_layout.format == format::bfzyx ||
         input_layout.format == format::bfzyx_f16 ||
         input_layout.format == format::bfzyx_b16f16) &&
        (input_layout.data_type == data_types::f32 || input_layout.data_type == data_types::f16) &&
        ((input_layout.size.feature[0] / conv->split()) % 16 == 0 || input_layout.size.feature[0] == 3) &&
        weights_layout.data_type == input_layout.data_type &&
        weights_layout.size.batch[0] % 16 == 0 &&
        conv->dilation == tensor(1) && conv->groups == 1)
        return true;
    return false;
}

bool layout_optimizer::convolution_fs_b_yx_fsv32_opt(layout const& input_layout,
                                                     layout const& weights_layout,
                                                     layout const& output_layout,
                                                     std::shared_ptr<const convolution> conv,
                                                     bool weak_restrictions) {
    // A set of rules that define when fs_b_yx_fsv32 mem format can be used
    bool correct_batch = output_layout.size.batch[0] != 1 && output_layout.size.batch[0] <= 16;
    bool correct_in_feature = input_layout.size.feature[0] >= 16;
    bool correct_out_feature = weak_restrictions ? output_layout.size.feature[0] >= 16 : output_layout.size.feature[0] > 16;
    bool dw_conv = static_cast<int>(conv->groups) == output_layout.size.feature[0];
    if (!correct_in_feature && input_layout.size.feature[0] == 3)  // bfyx with 3 feature -> fs_b_yx_fsv32 case
        correct_in_feature = true;

    if (input_layout.data_type != data_types::f16 || weights_layout.data_type != data_types::f16)
        return false;

    if ((input_layout.format == format::fs_b_yx_fsv32) || (correct_out_feature && correct_in_feature && correct_batch &&
        conv->split() == 1 && (dw_conv || conv->groups == 1) )) {
        return true;
    }
    return false;
}

bool layout_optimizer::deconvolution_bfzyx_f16_opt(layout const& input_layout,
    const layout& weights_layout,
    std::shared_ptr<const deconvolution> deconv) {
    // A set of rules that define when bfzyx_f16 mem format can be used
    if ((input_layout.format == format::bfzyx ||
            input_layout.format == format::bfzyx_f16 ||
            input_layout.format == format::bfzyx_b16f16) &&
        (input_layout.data_type == data_types::f32 ||
            input_layout.data_type == data_types::f16) &&
        weights_layout.size.batch[0] % 16 == 0 && input_layout.size.feature[0] % 16 == 0 &&
        deconv->split() == 1)
        return true;
    return false;
}

bool layout_optimizer::users_for_convolution_byxf_opt(program_node const& node, uint32_t depth) {
    // This function checks if byxf optimization can be applied to the required depth of node's users.
    // Setting depth to 1 will check only node's users, depth = 2 are user's users etc.
    if (depth == 0)
        return true;

    for (auto& user : node.get_users()) {
        // primitives that support transitions byxf->other format and other format->byxf are valid for byxf opt
        if (user->type() == cldnn::eltwise::type_id() || user->type() == cldnn::pooling::type_id()) {
            if (!users_for_convolution_byxf_opt(*user, depth - 1))
                return false;
        // convolution that is capable to use byxf and is performant is also valid for byxf opt
        } else if (user->type() == cldnn::convolution::type_id()) {
            if (convolution_byxf_opt(node.get_output_layout(),
                                     user->calc_output_layout(),
                                     user->get_dependency(1).get_output_layout(),
                                     user->as<convolution>())) {
                if (!users_for_convolution_byxf_opt(*user, depth - 1))
                    return false;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

bool layout_optimizer::deps_for_convolution_byxf_opt(program_node const& node, uint32_t depth) {
    // This function checks if requested format is the same for node's users in the required depth.
    // Setting depth to 1 will check only node's dependencies, depth = 2 are dep's dependencies etc.
    if (depth == 0)
        return true;

    for (auto& dep : node.get_dependencies()) {
        // skip data and generic_layers
        if (dep->is_type<data>() || dep->is_type<generic_layer>())
            continue;

        if (dep->is_type<convolution>()) {
            auto& conv_dep = dep->as<convolution>();
            if (!convolution_byxf_opt(conv_dep.input().get_output_layout(),
                                      conv_dep.get_output_layout(),
                                      conv_dep.weights().get_output_layout(),
                                      conv_dep)) {
                return false;
            }
        } else if (!dep->is_type<pooling>() && !dep->is_type<eltwise>()) {
            return false;
        }

        if (!deps_for_convolution_byxf_opt(*dep, depth - 1))
            return false;
    }
    return true;
}

format layout_optimizer::imad_case(convolution_node const& node) const {
    auto stride = node.get_primitive()->stride;
    auto out_size = node.get_output_layout().size;
    auto weights_dt = node.get_dependency(1).get_output_layout().data_type;

    auto dims_count = format::dimension(node.input().get_output_layout().format);

    bool is_grouped = node.get_split() > 1 || node.get_groups() > 1;
    bool is_dw = is_depthwise(node);

    if (dims_count == 5 && is_grouped) {
        return format::bfzyx;
    } else if (dims_count == 4 && is_grouped && !is_dw) {
        return format::bfyx;
    }

    bool asymmetric_quantization = node.activations_zero_points_term() || node.weights_zero_points_term();

    if (asymmetric_quantization && _optimization_attributes.b_fs_zyx_fsv32_network && out_size.feature[0] % 4 == 0) {
        if (dims_count == 5) {
            return format::b_fs_zyx_fsv32;
        } else {
            return format::b_fs_yx_fsv32;
        }
    }

    if (dims_count == 5) {
        return format::bfzyx;
    }

    if (stride.spatial[0] != stride.spatial[1] || out_size.spatial[0] != out_size.spatial[1] ||
        (weights_dt != data_types::u8 && weights_dt != data_types::i8)) {
        return format::byxf_af32;
    }

    if (node.get_groups() != 1) {  // depthwise parent
        // use byxf_af32 if there are any depthwise convolutions, else b_fs_yx_fsv4
        for (auto u : node.get_users()) {
            if (u->get_dependency(0).is_type<convolution>()) {
                convolution_node* conv_node = &u->get_dependency(0).as<convolution>();
                if (conv_node->get_groups() != 1)
                    return format::byxf_af32;
            }
        }
        return format::b_fs_yx_fsv4;
    } else {
        // use bxyf_af32 if every user is a depthwise convolution, else b_fs_yx_fsv4
        for (auto u : node.get_users()) {
            if (!u->get_dependency(0).is_type<convolution>())
                return format::b_fs_yx_fsv4;
            convolution_node *conv_node = &u->get_dependency(0).as<convolution>();
            if (conv_node->get_groups() == 1)
                return format::b_fs_yx_fsv4;
        }
        return format::byxf_af32;
    }
}

layout layout_optimizer::get_expected_layout(layout const& current_layout,
                                             convolution_node const& node,
                                             layout const& output_or_weights_layout) {
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;
    auto input_layout = node.get_dependency(0).get_output_layout();

    if ((input_layout.data_type == data_types::u8 || input_layout.data_type == data_types::i8)) {
        expected_format = imad_case(node);
        expected_tensor = current_layout.size;
    } else if (_optimization_attributes.bfzyx_f16_network &&
         convolution_bfzyx_f16_opt(input_layout,
                                   output_or_weights_layout, prim)) {
        expected_tensor = current_layout.size;
        if ((current_layout.data_type == data_types::f32 && expected_tensor.batch[0] % 16 == 0) ||
            (current_layout.data_type == data_types::f16 && expected_tensor.batch[0] % 32 == 0))
            expected_format = cldnn::format::bfzyx_b16f16;
        else
            expected_format = cldnn::format::bfzyx_f16;

    } else if (current_layout.format == format::bfzyx) {
        expected_tensor = current_layout.size;
        expected_format = cldnn::format::bfzyx;
    } else if ((_optimization_attributes.bfyx_f16_network &&
                convolution_bfyx_f16_opt(input_layout, output_or_weights_layout, prim)) ||
                input_layout.format == format::bfyx_f16) {
        expected_tensor = current_layout.size;
        expected_format = cldnn::format::bfyx_f16;
    } else if (current_layout.data_type == data_types::f16 &&
                layout_optimizer::convolution_byxf_opt(node.input().get_output_layout(), current_layout, output_or_weights_layout, node) &&
                (users_for_convolution_byxf_opt(node, 2) || deps_for_convolution_byxf_opt(node, 2)) &&
                // todo: remove this condition when yxfb optimizations will be disabled
                current_layout.format != cldnn::format::yxfb && current_layout.size.batch[0] == 1) {
        expected_tensor = current_layout.size;
        expected_format = cldnn::format::byxf;
    } else if (_optimization_attributes.fs_b_yx_fsv32_network && !node.get_transposed() &&
               ((convolution_fs_b_yx_fsv32_opt(node.get_dependency(0).get_output_layout(),
                                               node.get_dependency(1).get_output_layout(),
                                               node.get_output_layout(), prim) ||
               (node.get_dependency(0).is_type<convolution>() &&
                is_format_optimized(node.get_dependency(0).as<convolution>(), format::fs_b_yx_fsv32) &&
                convolution_fs_b_yx_fsv32_opt(node.get_dependency(0).get_output_layout(),
                                              node.get_dependency(1).get_output_layout(),
                                              node.get_output_layout(), prim, true))))) {
        // Chose fs_b_yx_fsv32 layout in two cases: 1-st: the current conv primitive totally supports fs_b_yx_fsv32 layout
        //                                          2-nd: the previous conv primitive supports fs_b_yx_fsv32 layout and
        //                                                current conv primitives supports this one with weak restrictions -
        //                                                that should be cheaper than reordering data to another layout
        expected_tensor = current_layout.size;
        expected_format = format::fs_b_yx_fsv32;
    } else if (current_layout.format == format::b_fs_yx_fsv4 ||
                current_layout.format == format::os_is_yx_osv16_isv4) {
        // imad case
        // nothing to do, just go out from here.
    } else if (layout_optimizer::convolution_bfyx_opt(current_layout, output_or_weights_layout, prim) ||
                (_output_size_handling_enabled && prim->with_output_size) || node.get_transposed()) {
        // commented out due to performance reasons, maybe enable in future
        /*if (current_layout.data_type == data_types::f32 &&
        current_layout.size.batch[0] % 16 == 0 &&
        current_layout.format == format::bfyx &&
        output_or_weights_layout.size.spatial[0] == 1 && output_or_weights_layout.size.spatial[1] == 1 &&
        prim->stride.spatial[0] == 1 && prim->stride.spatial[1] == 1 &&
        prim->input_offset.spatial[0] == 0 && prim->input_offset.spatial[1] == 0 &&
        !node.get_transposed())
    {
        if (!((current_layout.size.feature[0] % 8) == 0 && (current_layout.size.spatial[0] *
    current_layout.size.spatial[1]) == 16 && current_layout.data_padding == padding{ { 0,0,0,0 }, 0 }))
        {
            expected_tensor = current_layout.size.transform(cldnn::format::bf8_xy16, 1);
            expected_format = cldnn::format::bf8_xy16;
        }
    }
    else*/
        {
            expected_tensor = current_layout.size;
            if (current_layout.format == format::bfzyx_f16 || current_layout.format == format::bfzyx_b16f16)
                expected_format = cldnn::format::bfzyx;
            else
                expected_format = cldnn::format::bfyx;
        }

    } else {
        expected_tensor = current_layout.size;
        expected_format = cldnn::format::yxfb;
    }

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout,
                                             deconvolution_node const& node,
                                             layout const& output_or_weights_layout) {
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = current_layout.data_type;
    auto expected_format = current_layout.format;

    if (_optimization_attributes.bfzyx_f16_network &&
        deconvolution_bfzyx_f16_opt(current_layout, output_or_weights_layout, prim)) {
        expected_tensor = current_layout.size;
        if ((current_layout.data_type == data_types::f32 && expected_tensor.batch[0] % 16 == 0) ||
            (current_layout.data_type == data_types::f16 && expected_tensor.batch[0] % 32 == 0))
            expected_format = cldnn::format::bfzyx_b16f16;
        else
            expected_format = cldnn::format::bfzyx_f16;
    }
    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout,
                                             detection_output_node const& node,
                                             layout const& output_or_weights_layout) {
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = data_types::f32;
    auto expected_format = output_or_weights_layout.format;

    return layout(expected_data_type, expected_format, expected_tensor);
}

layout layout_optimizer::get_expected_layout(layout const& current_layout,
                                             binary_convolution_node const& node,
                                             layout const& /*output_or_weights_layout*/) {
    auto prim = node.get_primitive();
    auto expected_tensor = current_layout.size;
    auto expected_data_type = data_types::bin;
    auto expected_format = cldnn::format::b_fs_yx_32fp;

    return layout(expected_data_type, expected_format, expected_tensor);
}

format layout_optimizer::get_preferred_format(program_node& node) {
    format expected = format::any;
    auto output_layout = node.get_output_layout();

    if (!_format_forcing.empty() && _format_forcing.count(node.id()) != 0) {
        expected = _format_forcing.at(node.id());
    } else if (node.is_type<convolution>()) {
        auto& conv_node = node.as<convolution>();
        auto weights_layout = conv_node.weights(0).get_output_layout();
        expected = get_expected_layout(output_layout, conv_node, weights_layout).format;
    } else if (node.is_type<binary_convolution>()) {
        auto& bconv_node = node.as<binary_convolution>();
        auto weights_layout = bconv_node.weights(0).get_output_layout();
        expected = get_expected_layout(output_layout, bconv_node, weights_layout).format;
    } else if (node.is_type<detection_output>()) {
        expected = get_expected_layout(
            output_layout,
            node.as<detection_output>(),
            layout{ data_types::f32, format::bfyx, tensor{} }).format;
    } else if (node.is_type<reorder>() || node.is_type<input_layout>()) {
        expected = node.get_output_layout().format;
    } else if (node.is_type<deconvolution>()) {
        auto& deconv_node = node.as<deconvolution>();
        auto weights_layout = deconv_node.weights(0).get_output_layout();
        expected = get_expected_layout(output_layout, deconv_node, weights_layout).format;
    }

    return expected;
}

void layout_optimizer::set_optimization_attribute(optimization_attributes_type attribute, int32_t val) {
    switch (attribute) {
        case optimization_attributes_type::splitted_convolution:
            _optimization_attributes.splitted_convolution = val;
            break;
        case optimization_attributes_type::group_convolution:
            _optimization_attributes.group_convolution = val;
            break;
        case optimization_attributes_type::deformable_convolution:
            _optimization_attributes.deformable_convolution = val;
            break;
        case optimization_attributes_type::bfyx_only_layer:
            _optimization_attributes.bfyx_only_layer = val;
            break;
        case optimization_attributes_type::fs_b_yx_fsv32_network:
            _optimization_attributes.fs_b_yx_fsv32_network = val;
            break;
        case optimization_attributes_type::b_fs_zyx_fsv32_network:
            _optimization_attributes.b_fs_zyx_fsv32_network = val;
            break;
        case optimization_attributes_type::bfyx_f16_network:
            _optimization_attributes.bfyx_f16_network = val;
            break;
        case optimization_attributes_type::bfzyx_f16_network:
            _optimization_attributes.bfzyx_f16_network = val;
            break;
        default:
            throw std::out_of_range("unsupported layout optimization attribute");
    }
}

bool layout_optimizer::is_format_optimized(const convolution_node& node, const format& format) {
    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights().get_output_layout();
    auto prim = node.get_primitive();

    switch (format) {
        case format::bfyx_f16:
            return convolution_bfyx_f16_opt(input_layout, weights_layout, prim) &&
                   // Work-around for inability to use bfyx_f16 and winograd together
                   !should_use_winograd_2x3_s1(prim, input_layout, weights_layout, _output_size_handling_enabled);
        case format::bfzyx_f16:
        case format::bfzyx_b16f16:
            return convolution_bfzyx_f16_opt(input_layout, weights_layout, prim);
        case format::fs_b_yx_fsv32:
            return convolution_fs_b_yx_fsv32_opt(input_layout, weights_layout, node.get_output_layout(), prim);
        default:
            throw std::invalid_argument(
                "[Layout optimizer] Other formats in is_format_optimized(...) method are not implemented!");
    }
}

bool layout_optimizer::is_format_optimized(const deconvolution_node& node, const format& format) {
    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights().get_output_layout();
    auto prim = node.get_primitive();

    switch (format) {
    case format::bfzyx_f16:
    case format::bfzyx_b16f16:
        return deconvolution_bfzyx_f16_opt(input_layout, weights_layout, prim);
    default:
        throw std::invalid_argument(
            "[Layout optimizer] Other formats in is_format_optimized(...) method are not implemented!");
    }
}

void layout_optimizer::set_implementation_forcing(const implementation_forcing_map& map) {
    for (const auto& kv : map) {
        _format_forcing.emplace(kv.first, kv.second.output_format);
    }
}
