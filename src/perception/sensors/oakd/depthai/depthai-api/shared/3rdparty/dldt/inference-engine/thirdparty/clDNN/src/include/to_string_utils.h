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
#pragma once
#include <string>
#include <sstream>
#include "api/tensor.hpp"
#include "api/layout.hpp"
#include "api/primitive.hpp"
#include <memory>

namespace cldnn {

inline std::string to_string_hex(int val) {
    std::stringstream stream;
    stream << "0x" << std::uppercase << std::hex << val;
    return stream.str();
}

inline std::string bool_to_str(bool cond) { return cond ? "true" : "false"; }

inline std::string get_extr_type(const std::string& str) {
    auto begin = str.find('<');
    auto end = str.find('>');

    if (begin == std::string::npos || end == std::string::npos)
        return {};

    return str.substr(begin + 1, (end - begin) - 1);
}

inline std::string dt_to_str(data_types dt) {
    switch (dt) {
        case data_types::bin:
            return "bin";
        case data_types::i8:
            return "i8";
        case data_types::u8:
            return "u8";
        case data_types::i32:
            return "i32";
        case data_types::i64:
            return "i64";
        case data_types::f16:
            return "f16";
        case data_types::f32:
            return "f32";
        default:
            return "unknown (" + std::to_string(typename std::underlying_type<data_types>::type(dt)) + ")";
    }
}

inline std::string fmt_to_str(format fmt) {
    switch (fmt.value) {
        case format::yxfb:
            return "yxfb";
        case format::byxf:
            return "byxf";
        case format::bfyx:
            return "bfyx";
        case format::fyxb:
            return "fyxb";
        case format::bfyx_f16:
            return "bfyx_f16";
        case format::b_fs_yx_fsv32:
            return "b_fs_yx_fsv32";
        case format::b_fs_zyx_fsv32:
            return "b_fs_zyx_fsv32";
        case format::bs_xs_xsv8_bsv8:
            return "bs_xs_xsv8_bsv8";
        case format::bs_xs_xsv8_bsv16:
            return "bs_xs_xsv8_bsv16";
        case format::bs_x_bsv16:
            return "bs_x_bsv16";
        case format::bf8_xy16:
            return "bf8_xy16";
        case format::winograd_2x3_s1_data:
            return "winograd_2x3_s1_data";
        case format::byxf_af32:
            return "byxf_af32";
        case format::byx8_f4:
            return "byx8_f4";
        case format::fs_bs_yx_bsv4_fsv32:
            return "fs_bs_yx_bsv4_fsv32";
        case format::b_fs_yx_fsv4:
            return "b_fs_yx_fsv4";
        case format::b_fs_yx_32fp:
            return "b_fs_yx_32fp";
        case format::bfzyx:
            return "bfzyx";
        case format::bfwzyx:
            return "bfwzyx";
        case format::fs_b_yx_fsv32:
            return "fs_b_yx_fsv32";
        case format::bfzyx_f16:
            return "bfzyx_f16";
        case format::bfzyx_b16f16:
            return "bfzyx_b16f16";

        case format::winograd_2x3_s1_weights:
            return "winograd_2x3_s1_weights";
        case format::winograd_2x3_s1_fused_weights:
            return "winograd_2x3_s1_fused_weights";
        case format::winograd_6x3_s1_fused_weights:
            return "winograd_6x3_s1_fused_weights";
        case format::image_2d_weights_c4_fyx_b:
            return "image_2d_weights_c4_fyx_b";
        case format::image_2d_weights_c1_b_fyx:
            return "image_2d_weights_c1_b_fyx";
        case format::image_2d_weights_winograd_6x3_s1_fbxyb:
            return "image_2d_weights_winograd_6x3_s1_fbxyb";
        case format::image_2d_weights_winograd_6x3_s1_xfbyb:
            return "image_2d_weights_winograd_6x3_s1_xfbyb";
        case format::oiyx_o16:
            return "oiyx_o16";
        case format::os_iyx_osv16:
            return "os_iyx_osv16";
        case format::os_iyx_osv32:
            return "os_iyx_osv32";
        case format::os_iyx_osv64:
            return "os_iyx_osv64";
        case format::is_o_yx_isv32:
            return "is_o_yx_isv32";
        case format::o_i_yx_i16_o16:
            return "o_i_yx_i16_o16";
        case format::os_is_yx_isa8_osv8_isv4:
            return "os_is_yx_isa8_osv8_isv4";
        case format::os_is_zyx_isa8_osv8_isv4:
            return "os_is_zyx_isa8_osv8_isv4";
        case format::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4";
        case format::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4";
        case format::os_is_yx_isa8_osv8_isv4_swizzled_by_4:
            return "os_is_yx_isa8_osv8_isv4_swizzled_by_4";
        case format::is_o32_yx_isv32_swizzled_by_4:
            return "is_o32_yx_isv32_swizzled_by_4";
        case format::bf_lyx_yx:
            return "bf_lyx_yx";
        case format::os_is_yx_osv16_isv4:
            return "os_is_yx_osv16_isv4";
        case format::os_is_yx_osv32_isv4_swizzled_by_2:
            return "os_is_yx_osv32_isv4_swizzled_by_2";
        case format::os_is_y_x8_osv8_isv4:
            return "os_is_y_x8_osv8_isv4";
        case format::os_is_yx_osv32_isv32p:
            return "os_is_yx_osv32_isv32p";
        case format::o_i_zyx_i16_o16:
            return "o_i_zyx_i16_o16";
        case format::i_o_zyx_o16_i16:
            return "i_o_zyx_o16_i16";
        case format::os_is_osv32_isv32_swizzled_by_4:
            return "os_is_osv32_isv32_swizzled_by_4";
        case format::o_i_zyx_i8_o16_i2:
            return "o_i_zyx_i8_o16_i2";
        case format::ozyxi_o16:
            return "ozyxi_o16";
        default:
            return "unknown (" + std::to_string(fmt.value) + ")";
    }
}

inline std::string type_to_str(std::shared_ptr<const primitive> primitive) { return primitive->type_string(); }

}  // namespace cldnn
