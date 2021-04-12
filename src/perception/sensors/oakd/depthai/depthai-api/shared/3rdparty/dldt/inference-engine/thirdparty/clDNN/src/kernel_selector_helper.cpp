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

#include "kernel_selector_helper.h"
#include "kernel_selector_params.h"
#include "to_string_utils.h"

#include "gpu/ocl_toolkit.h"

#include "program_node.h"
#include "program_impl.h"

#include "training_params.h"
#include <string>
#include <vector>

kernel_selector::data_type to_data_type(data_types dt) {
    switch (dt) {
        case cldnn::data_types::bin:
            return kernel_selector::data_type::BINARY;
        case cldnn::data_types::i8:
            return kernel_selector::data_type::INT8;
        case cldnn::data_types::u8:
            return kernel_selector::data_type::UINT8;
        case cldnn::data_types::i32:
            return kernel_selector::data_type::INT32;
        case cldnn::data_types::i64:
            return kernel_selector::data_type::INT64;
        case cldnn::data_types::f16:
            return kernel_selector::data_type::F16;
        case cldnn::data_types::f32:
            return kernel_selector::data_type::F32;
        default:
            assert(0);
            return kernel_selector::data_type::F16;
    }
}

data_types from_data_type(kernel_selector::data_type dt) {
    switch (dt) {
        case kernel_selector::data_type::BINARY:
            return cldnn::data_types::bin;
        case kernel_selector::data_type::INT8:
            return cldnn::data_types::i8;
        case kernel_selector::data_type::UINT8:
            return cldnn::data_types::u8;
        case kernel_selector::data_type::INT32:
            return cldnn::data_types::i32;
        case kernel_selector::data_type::INT64:
            return cldnn::data_types::i64;
        case kernel_selector::data_type::F16:
            return cldnn::data_types::f16;
        case kernel_selector::data_type::F32:
            return cldnn::data_types::f32;
        default:
            assert(0);
            return cldnn::data_types::f16;
    }
}

kernel_selector::weights_type to_weights_type(data_types dt) {
    switch (dt) {
        case cldnn::data_types::bin:
            return kernel_selector::weights_type::BINARY;
        case cldnn::data_types::i8:
            return kernel_selector::weights_type::INT8;
        case cldnn::data_types::u8:
            return kernel_selector::weights_type::UINT8;
        case cldnn::data_types::f16:
            return kernel_selector::weights_type::F16;
        case cldnn::data_types::f32:
            return kernel_selector::weights_type::F32;
        default:
            assert(0);
            return kernel_selector::weights_type::F16;
    }
}

data_types from_weights_type(kernel_selector::weights_type dt) {
    switch (dt) {
        case kernel_selector::weights_type::BINARY:
            return data_types::bin;
        case kernel_selector::weights_type::INT8:
            return data_types::i8;
        case kernel_selector::weights_type::UINT8:
            return data_types::u8;
        case kernel_selector::weights_type::F16:
            return data_types::f16;
        case kernel_selector::weights_type::F32:
            return data_types::f32;
        default:
            assert(0);
            return data_types::f16;
    }
}

kernel_selector::data_layout to_data_layout(format f) {
    switch (f) {
        case format::bfyx:
            return kernel_selector::data_layout::bfyx;
        case format::yxfb:
            return kernel_selector::data_layout::yxfb;
        case format::byxf:
            return kernel_selector::data_layout::byxf;
        case format::fyxb:
            return kernel_selector::data_layout::fyxb;
        case format::bfyx_f16:
            return kernel_selector::data_layout::bfyx_f16;
        case format::b_fs_yx_fsv32:
            return kernel_selector::data_layout::b_fs_yx_fsv32;
        case format::b_fs_zyx_fsv32:
            return kernel_selector::data_layout::b_fs_zyx_fsv32;
        case format::bs_x_bsv16:
            return kernel_selector::data_layout::bs_f_bsv16__af8;
        case format::bs_xs_xsv8_bsv8:
            return kernel_selector::data_layout::bs_f_bsv8__af8;
        case format::bs_xs_xsv8_bsv16:
            return kernel_selector::data_layout::bs_f_bsv16__af8;
        case format::bf8_xy16:
            return kernel_selector::data_layout::bf8_xy16;
        case format::winograd_2x3_s1_data:
            return kernel_selector::data_layout::winograd_2x3_s1_data;
        case format::byxf_af32:
            return kernel_selector::data_layout::byxf_af32;
        case format::byx8_f4:
            return kernel_selector::data_layout::byx8_f4;
        case format::fs_bs_yx_bsv4_fsv32:
            return kernel_selector::data_layout::fs_bs_yx_bsv4_fsv32;
        case format::b_fs_yx_fsv4:
            return kernel_selector::data_layout::b_fs_yx_fsv4;
        case format::b_fs_yx_32fp:
            return kernel_selector::data_layout::b_fs_yx_32fp;
        case format::bfzyx:
            return kernel_selector::data_layout::bfzyx;
        case format::fs_b_yx_fsv32:
            return kernel_selector::data_layout::fs_b_yx_fsv32;
        case format::bfwzyx:
            return kernel_selector::data_layout::bfwzyx;
        case format::bfzyx_f16:
            return kernel_selector::data_layout::bfzyx_f16;
        case format::bfzyx_b16f16:
            return kernel_selector::data_layout::bfzyx_b16f16;
        case format::nv12:
            return kernel_selector::data_layout::nv12;
        default:
            throw std::invalid_argument("Format f is not a proper data layout");
    }
}

cldnn::format from_data_layout(kernel_selector::data_layout l) {
    switch (l) {
        case kernel_selector::data_layout::bf:
            return cldnn::format::bfyx;
        case kernel_selector::data_layout::fb:
            return cldnn::format::fyxb;
        case kernel_selector::data_layout::bfyx:
            return cldnn::format::bfyx;
        case kernel_selector::data_layout::yxfb:
            return cldnn::format::yxfb;
        case kernel_selector::data_layout::byxf:
            return cldnn::format::byxf;
        case kernel_selector::data_layout::fyxb:
            return cldnn::format::fyxb;
        case kernel_selector::data_layout::bfyx_f16:
            return cldnn::format::bfyx_f16;
        case kernel_selector::data_layout::b_fs_yx_fsv32:
            return cldnn::format::b_fs_yx_fsv32;
        case kernel_selector::data_layout::b_fs_zyx_fsv32:
            return cldnn::format::b_fs_zyx_fsv32;
        case kernel_selector::data_layout::bs_f_bsv8__af8:
            return cldnn::format::bs_xs_xsv8_bsv8;
        case kernel_selector::data_layout::bs_f_bsv16__af8:
            return cldnn::format::bs_x_bsv16;
        case kernel_selector::data_layout::bf8_xy16:
            return cldnn::format::bf8_xy16;
        case kernel_selector::data_layout::winograd_2x3_s1_data:
            return cldnn::format::winograd_2x3_s1_data;
        case kernel_selector::data_layout::byxf_af32:
            return cldnn::format::byxf_af32;
        case kernel_selector::data_layout::byx8_f4:
            return cldnn::format::byx8_f4;
        case kernel_selector::data_layout::fs_bs_yx_bsv4_fsv32:
            return cldnn::format::fs_bs_yx_bsv4_fsv32;
        case kernel_selector::data_layout::b_fs_yx_32fp:
            return cldnn::format::b_fs_yx_32fp;
        case kernel_selector::data_layout::bfzyx:
            return cldnn::format::bfzyx;
        case kernel_selector::data_layout::fs_b_yx_fsv32:
            return cldnn::format::fs_b_yx_fsv32;
        case kernel_selector::data_layout::bfwzyx:
            return cldnn::format::bfwzyx;
        case kernel_selector::data_layout::b_fs_yx_fsv4:
            return cldnn::format::b_fs_yx_fsv4;
        case kernel_selector::data_layout::nv12:
            return cldnn::format::nv12;
        default:
            throw std::invalid_argument("Unable to convert data layout " + std::to_string(l) + " to tensor format");
    }
}

kernel_selector::weights_layout to_weights_layout(format f) {
    switch (f) {
        case format::bfyx:
            return kernel_selector::weights_layout::oiyx;
        case format::fyxb:
            return kernel_selector::weights_layout::iyxo;
        case format::byxf:
            return kernel_selector::weights_layout::oyxi;
        case format::yxfb:
            return kernel_selector::weights_layout::yxio;
        case format::os_iyx_osv16:
            return kernel_selector::weights_layout::os_iyx_osv16;
        case format::os_iyx_osv32:
            return kernel_selector::weights_layout::os_iyx_osv32;
        case format::os_iyx_osv64:
            return kernel_selector::weights_layout::os_iyx_osv64;
        case format::image_2d_weights_c4_fyx_b:
            return kernel_selector::weights_layout::image_2d_weights_c4_fyx_b;
        case format::image_2d_weights_c1_b_fyx:
            return kernel_selector::weights_layout::image_2d_weights_c1_b_fyx;
        case format::winograd_2x3_s1_weights:
            return kernel_selector::weights_layout::winograd_2x3_s1_weights;
        case format::winograd_2x3_s1_fused_weights:
            return kernel_selector::weights_layout::winograd_2x3_s1_fused_weights;
        case format::winograd_6x3_s1_fused_weights:
            return kernel_selector::weights_layout::winograd_6x3_s1_fused_weights;
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            return kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_fbxyb;
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            return kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_xfbyb;
        case format::os_is_yx_isa8_osv8_isv4:
            return kernel_selector::weights_layout::os_is_yx_isa8_osv8_isv4;
        case format::os_is_zyx_isa8_osv8_isv4:
            return kernel_selector::weights_layout::os_is_zyx_isa8_osv8_isv4;
        case format::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return kernel_selector::weights_layout::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4;
        case format::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return kernel_selector::weights_layout::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4;
        case format::os_is_yx_isa8_osv8_isv4_swizzled_by_4:
            return kernel_selector::weights_layout::os_is_yx_isa8_osv8_isv4_swizzled_by_4;
        case format::is_o_yx_isv32:
            return kernel_selector::weights_layout::is_o_yx_isv32;
        case format::is_o32_yx_isv32_swizzled_by_4:
            return kernel_selector::weights_layout::is_o32_yx_isv32_swizzled_by_4;
        case format::os_is_y_x8_osv8_isv4:
            return kernel_selector::weights_layout::os_is_y_x8_osv8_isv4;
        case format::os_is_yx_osv16_isv4:
            return kernel_selector::weights_layout::os_is_yx_osv16_isv4;
        case format::os_is_yx_osv32_isv4_swizzled_by_2:
            return kernel_selector::weights_layout::os_is_yx_osv32_isv4_swizzled_by_2;
        case format::os_is_yx_osv32_isv32p:
            return kernel_selector::weights_layout::os_is_yx_osv32_isv32p;
        case format::o_i_yx_i16_o16:
            return kernel_selector::weights_layout::o_i_yx_i16_o16;
        case format::os_is_y_x8_osv8_isv4_swizzled_by_4:
            return kernel_selector::weights_layout::os_is_y_x8_osv8_isv4_swizzled_by_4;
        case format::bfzyx:
            return kernel_selector::weights_layout::oizyx;
        case format::bs_xs_xsv8_bsv8:
            return kernel_selector::weights_layout::os_i_osv8__ai8;
        case format::bs_xs_xsv8_bsv16:
            return kernel_selector::weights_layout::os_i_osv16__ai8;
        case format::bs_x_bsv16:
            return kernel_selector::weights_layout::os_i_osv16;
        case format::bf_lyx_yx:
            return kernel_selector::weights_layout::bf_lyx_yx;
        case format::oiyx_o16:
            return kernel_selector::weights_layout::oiyx_o16;
        case format::o_i_zyx_i16_o16:
            return kernel_selector::weights_layout::o_i_zyx_i16_o16;
        case format::i_o_zyx_o16_i16:
            return kernel_selector::weights_layout::i_o_zyx_o16_i16;
        case format::os_is_osv32_isv32_swizzled_by_4:
            return kernel_selector::weights_layout::os_is_osv32_isv32_swizzled_by_4;
        case format::o_i_zyx_i8_o16_i2:
            return kernel_selector::weights_layout::o_i_zyx_i8_o16_i2;
        case format::ozyxi_o16:
            return kernel_selector::weights_layout::ozyxi_o16;
        default:
            throw std::invalid_argument("Unable to convert tensor layout " + fmt_to_str(f) + " to weights layout");
    }
}

cldnn::format::type from_weights_layout(kernel_selector::weights_layout l) {
    switch (l) {
        case kernel_selector::weights_layout::oi:
            return cldnn::format::bfyx;
        case kernel_selector::weights_layout::oiyx:
            return cldnn::format::bfyx;
        case kernel_selector::weights_layout::oyxi:
            return cldnn::format::byxf;
        case kernel_selector::weights_layout::io:
            return cldnn::format::fyxb;
        case kernel_selector::weights_layout::iyxo:
            return cldnn::format::fyxb;
        case kernel_selector::weights_layout::yxio:
            return cldnn::format::yxfb;
        case kernel_selector::weights_layout::os_iyx_osv16:
            return cldnn::format::os_iyx_osv16;
        case kernel_selector::weights_layout::os_iyx_osv32:
            return cldnn::format::os_iyx_osv32;
        case kernel_selector::weights_layout::os_iyx_osv64:
            return cldnn::format::os_iyx_osv64;
        case kernel_selector::weights_layout::os_i_osv16:
            return cldnn::format::bs_x_bsv16;
        case kernel_selector::weights_layout::os_i_osv8__ai8:
            return cldnn::format::bs_xs_xsv8_bsv8;
        case kernel_selector::weights_layout::os_i_osv16__ai8:
            return cldnn::format::bs_xs_xsv8_bsv16;
        case kernel_selector::weights_layout::image_2d_weights_c4_fyx_b:
            return cldnn::format::image_2d_weights_c4_fyx_b;
        case kernel_selector::weights_layout::image_2d_weights_c1_b_fyx:
            return cldnn::format::image_2d_weights_c1_b_fyx;
        case kernel_selector::weights_layout::winograd_2x3_s1_weights:
            return cldnn::format::winograd_2x3_s1_weights;
        case kernel_selector::weights_layout::winograd_2x3_s1_fused_weights:
            return cldnn::format::winograd_2x3_s1_fused_weights;
        case kernel_selector::weights_layout::winograd_6x3_s1_fused_weights:
            return cldnn::format::winograd_6x3_s1_fused_weights;
        case kernel_selector::weights_layout::oiyx_o16:
            return cldnn::format::oiyx_o16;
        case kernel_selector::weights_layout::o_i_yx_i16_o16:
            return cldnn::format::o_i_yx_i16_o16;
        case kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_fbxyb:
            return cldnn::format::image_2d_weights_winograd_6x3_s1_fbxyb;
        case kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_xfbyb:
            return cldnn::format::image_2d_weights_winograd_6x3_s1_xfbyb;
        case kernel_selector::weights_layout::os_is_yx_isa8_osv8_isv4:
            return cldnn::format::os_is_yx_isa8_osv8_isv4;
        case kernel_selector::weights_layout::os_is_zyx_isa8_osv8_isv4:
            return cldnn::format::os_is_zyx_isa8_osv8_isv4;
        case kernel_selector::weights_layout::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return cldnn::format::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4;
        case kernel_selector::weights_layout::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return cldnn::format::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4;
        case kernel_selector::weights_layout::os_is_yx_isa8_osv8_isv4_swizzled_by_4:
            return cldnn::format::os_is_yx_isa8_osv8_isv4_swizzled_by_4;
        case kernel_selector::weights_layout::is_o_yx_isv32:
            return cldnn::format::is_o_yx_isv32;
        case kernel_selector::weights_layout::is_o32_yx_isv32_swizzled_by_4:
            return cldnn::format::is_o32_yx_isv32_swizzled_by_4;
        case kernel_selector::weights_layout::os_is_y_x8_osv8_isv4:
            return cldnn::format::os_is_y_x8_osv8_isv4;
        case kernel_selector::weights_layout::os_is_yx_osv32_isv4_swizzled_by_2:
            return format::os_is_yx_osv32_isv4_swizzled_by_2;
        case kernel_selector::weights_layout::os_is_y_x8_osv8_isv4_swizzled_by_4:
            return cldnn::format::os_is_y_x8_osv8_isv4_swizzled_by_4;
        case kernel_selector::weights_layout::bf_lyx_yx:
            return cldnn::format::bf_lyx_yx;
        case kernel_selector::weights_layout::os_is_yx_osv32_isv32p:
            return cldnn::format::os_is_yx_osv32_isv32p;
        case kernel_selector::weights_layout::oizyx:
            return cldnn::format::bfzyx;
        case kernel_selector::weights_layout::o_i_zyx_i16_o16:
            return cldnn::format::o_i_zyx_i16_o16;
        case kernel_selector::weights_layout::i_o_zyx_o16_i16:
            return cldnn::format::i_o_zyx_o16_i16;
        case kernel_selector::weights_layout::o_i_zyx_i8_o16_i2:
            return cldnn::format::o_i_zyx_i8_o16_i2;
        case kernel_selector::weights_layout::ozyxi_o16:
            return cldnn::format::ozyxi_o16;
        default:
            return cldnn::format::bfyx;
    }
}

kernel_selector::tuning_mode to_tuning_mode(cldnn::tuning_mode mode) {
    switch (mode) {
        case cldnn::tuning_mode::tuning_disabled:
            return kernel_selector::tuning_mode::TUNING_DISABLED;
        case cldnn::tuning_mode::tuning_use_cache:
            return kernel_selector::tuning_mode::TUNING_USE_CACHE;
        case cldnn::tuning_mode::tuning_tune_and_cache:
            return kernel_selector::tuning_mode::TUNING_TUNE_AND_CACHE;
        default:
            return kernel_selector::tuning_mode::TUNING_DISABLED;
    }
}

std::string to_host_version(const cldnn::version_t& version) {
    std::stringstream ss;
    ss << version.major << "." << version.minor << "." << version.build << "." << version.revision;
    return ss.str();
}

kernel_selector::data_tensor convert_data_tensor(const layout& l, uint32_t split, const tensor view_offset) {
    const auto& pad = l.data_padding;
    const auto& vals = l.size.sizes(l.format);
    const auto& add_offsets = view_offset.sizes(l.format);
    const auto& lower_pad = pad.lower_size().sizes(l.format);
    const auto& upper_pad = pad.upper_size().sizes(l.format);
    const auto ks_layout = to_data_layout(l.format);
    kernel_selector::n_dims vec(kernel_selector::DataTensor::ChannelsCount(ks_layout));

    size_t pitch = 1;
    size_t offset = 0;

    auto new_vals = vals;

    if (ks_layout == kernel_selector::Tensor::byxf_af32) {
        new_vals[3] = align_to(vals[3], 32);
    }
    if (ks_layout == kernel_selector::Tensor::b_fs_yx_fsv32) {
        new_vals[1] = align_to(vals[1], 32);
    }
    if (ks_layout == kernel_selector::Tensor::fs_bs_yx_bsv4_fsv32) {
        new_vals[3] = align_to(vals[3], 32);
        new_vals[2] = align_to(vals[2], 4);
    }
    if (ks_layout == kernel_selector::Tensor::byx8_f4) {
        new_vals[3] = align_to(vals[3], 4);
        new_vals[2] = align_to(vals[2], 8);
    }

    for (size_t i = 0; i < vec.size(); i++) {
        const size_t tensor_index = vec.size() - 1 - i;
        const auto d = vals[tensor_index];
        const auto lp = lower_pad[tensor_index];
        const auto up = upper_pad[tensor_index];
        // tells us how many elements are reserved in memory for this tensor index
        const auto reserved_in_mem_count = new_vals[tensor_index];

        auto& elm = vec[i];
        elm.v = static_cast<size_t>(d - add_offsets[tensor_index]);
        elm.pitch = pitch;
        elm.pad.before = lp;
        elm.pad.after = up;

        offset += pitch * (add_offsets[tensor_index]);
        pitch *= (reserved_in_mem_count + lp + up);
    }

    const int feature_index =
        kernel_selector::DataTensor::Channelndex(ks_layout, kernel_selector::Tensor::DataChannelName::FEATURE);
    vec[feature_index].v /= split;

    return kernel_selector::data_tensor(vec, to_data_type(l.data_type), ks_layout, offset);
}

kernel_selector::weights_tensor convert_weights_tensor(const layout& l) {
    const auto& t = l.size.sizes(l.format);
    const auto base_layout = to_weights_layout(l.format);
    const auto ks_type = to_weights_type(l.data_type);
    const auto ks_layout = to_weights_layout(l.format);
    std::vector<size_t> vec(kernel_selector::WeightsTensor::ChannelsCount(base_layout));

    for (size_t i = 0; i < vec.size(); i++) {
        const size_t tensor_index = t.size() - 1 - i;
        const auto d = t[tensor_index];
        vec[i] = static_cast<size_t>(d);
    }

    return kernel_selector::weights_tensor(vec, ks_type, base_layout).TransformIgnorePadding(ks_layout);
}

layout from_weights_tensor(const kernel_selector::weights_tensor& l) {
    const auto format = from_weights_layout(l.GetLayout());
    const auto type = from_weights_type(l.GetDType());

    tensor t = {static_cast<int>(l.OFM().v),
                static_cast<int>(l.IFM().v),
                static_cast<int>(l.X().v),
                static_cast<int>(l.Y().v),
                static_cast<int>(l.LX().v),
                static_cast<int>(l.LY().v)};

    return layout(type, format, t);
}

kernel_selector::activation_function get_kernel_selector_activation_param(activation_func activation) {
    switch (activation) {
        case cldnn::activation_func::none:
            return kernel_selector::activation_function::NONE;
        case cldnn::activation_func::logistic:
            return kernel_selector::activation_function::LOGISTIC;
        case cldnn::activation_func::hyperbolic_tan:
            return kernel_selector::activation_function::HYPERBOLIC_TAN;
        case cldnn::activation_func::relu:
            return kernel_selector::activation_function::RELU;
        case cldnn::activation_func::relu_negative_slope:
            return kernel_selector::activation_function::RELU_NEGATIVE_SLOPE;
        case cldnn::activation_func::clamp:
            return kernel_selector::activation_function::CLAMP;
        case cldnn::activation_func::softrelu:
            return kernel_selector::activation_function::SOFTRELU;
        case cldnn::activation_func::abs:
            return kernel_selector::activation_function::ABS;
        case cldnn::activation_func::linear:
            return kernel_selector::activation_function::LINEAR;
        case cldnn::activation_func::square:
            return kernel_selector::activation_function::SQUARE;
        case cldnn::activation_func::sqrt:
            return kernel_selector::activation_function::SQRT;
        case cldnn::activation_func::elu:
            return kernel_selector::activation_function::ELU;
        case cldnn::activation_func::sin:
            return kernel_selector::activation_function::SIN;
        case cldnn::activation_func::asin:
            return kernel_selector::activation_function::ASIN;
        case cldnn::activation_func::sinh:
            return kernel_selector::activation_function::SINH;
        case cldnn::activation_func::asinh:
            return kernel_selector::activation_function::ASINH;
        case cldnn::activation_func::cos:
            return kernel_selector::activation_function::COS;
        case cldnn::activation_func::acos:
            return kernel_selector::activation_function::ACOS;
        case cldnn::activation_func::cosh:
            return kernel_selector::activation_function::COSH;
        case cldnn::activation_func::acosh:
            return kernel_selector::activation_function::ACOSH;
        case cldnn::activation_func::log:
            return kernel_selector::activation_function::LOG;
        case cldnn::activation_func::log2:
            return kernel_selector::activation_function::LOG2;
        case cldnn::activation_func::exp:
            return kernel_selector::activation_function::EXP;
        case cldnn::activation_func::tan:
            return kernel_selector::activation_function::TAN;
        case cldnn::activation_func::atan:
            return kernel_selector::activation_function::ATAN;
        case cldnn::activation_func::atanh:
            return kernel_selector::activation_function::ATANH;
        case cldnn::activation_func::floor:
            return kernel_selector::activation_function::FLOOR;
        case cldnn::activation_func::ceil:
            return kernel_selector::activation_function::CEIL;
        case cldnn::activation_func::negative:
            return kernel_selector::activation_function::NEGATIVE;
        case cldnn::activation_func::negation:
            return kernel_selector::activation_function::NOT;
        case cldnn::activation_func::pow:
            return kernel_selector::activation_function::POW;
        case cldnn::activation_func::erf:
            return kernel_selector::activation_function::ERF;
        case cldnn::activation_func::reciprocal:
            return kernel_selector::activation_function::RECIPROCAL;
        case cldnn::activation_func::selu:
            return kernel_selector::activation_function::SELU;
        case cldnn::activation_func::sign:
            return kernel_selector::activation_function::SIGN;
        case cldnn::activation_func::softplus:
            return kernel_selector::activation_function::SOFTPLUS;
        case cldnn::activation_func::softsign:
            return kernel_selector::activation_function::SOFTSIGN;
        case cldnn::activation_func::hard_sigmoid:
            return kernel_selector::activation_function::HARD_SIGMOID;
        default:
            throw std::runtime_error("Unknown activation function");
            break;
    }
}

kernel_selector::activation_function get_kernel_selector_activation_grad_param(
    activation_grad_func activation_grad_func) {
    switch (activation_grad_func) {
        case cldnn::activation_grad_func::none:
            return kernel_selector::activation_function::NONE_GRAD;
        case cldnn::activation_grad_func::relu:
            return kernel_selector::activation_function::RELU_GRAD;
        case cldnn::activation_grad_func::relu_negative_slope:
            return kernel_selector::activation_function::RELU_NEGATIVE_SLOPE_GRAD;
        default:
            throw std::runtime_error("Unknown activation_grad function");
            break;
    }
}

void set_params(const program_node& node, kernel_selector::params& params) {
    const auto& program = node.get_program();
    const auto& context = program.get_engine().get_context();
    const auto& device_info = context->get_device_info();

    params.engineInfo.bSubGroupSupport = context->extension_supported("cl_intel_subgroups");
    params.engineInfo.bSubGroupShortSupport = context->extension_supported("cl_intel_subgroups_short");
    params.engineInfo.bFP16Support = context->extension_supported("cl_khr_fp16");
    params.engineInfo.bFP64Support = context->extension_supported("cl_khr_fp64");
    params.engineInfo.bIMADSupport = device_info.supports_imad != 0;
    params.engineInfo.bIMMADSupport = device_info.supports_immad != 0;
    params.engineInfo.bImageSupport = device_info.supports_image != 0;
    params.engineInfo.maxWorkGroupSize = device_info.max_work_group_size;
    params.engineInfo.maxLocalMemSize = device_info.max_local_mem_size;
    params.engineInfo.maxImage2dWidth = device_info.max_image2d_width;
    params.engineInfo.maxImage2dHeight = device_info.max_image2d_height;
    params.engineInfo.computeUnitsCount = device_info.compute_units_count;
    params.engineInfo.deviceCache = context->get_device_cache();
    params.engineInfo.driverVersion = device_info.driver_version;
    params.engineInfo.hostVersion = to_host_version(cldnn::get_version());

    auto impl_forcing_bo = program.get_options().get<build_option_type::force_implementations>();
    const auto& impl_forcing = impl_forcing_bo->forcing;

    if (impl_forcing.count(node.id()) != 0) {
        params.forceImplementation = impl_forcing.at(node.id()).kernel_name;
    }
}

void set_learning_params(const program_node& node, kernel_selector::training_params& params, bool use_momentum) {
    const auto learning_params =
        node.get_program().get_options().template get<build_option_type::learning_config>()->params;

    if (use_momentum) {
        params.use_momentum = true;
    }

    params.momentum_factor = learning_params.momentum;
    params.weights_decay = learning_params.weights_decay;
}

void set_optional_params(const program_impl& program, kernel_selector::optional_params& params) {
    const auto& context = program.get_engine().get_context();

    params.meaningfulKernelsNames = context->get_configuration().meaningful_kernels_names;
    params.allowStaticInputReordering = program.get_options().get<build_option_type::optimize_data>()->enabled();
    params.allowInputReordering = false;
    params.allowOutputReordering = false;

    const auto& tuning_config = program.get_options().get<build_option_type::tuning_config>();
    params.tuningParams.mode = to_tuning_mode(tuning_config->config.mode);
    params.tuningParams.cacheFilePath = tuning_config->config.cache_file_path;
}
