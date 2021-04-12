/*
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

#include <cstddef>
#include "tensor_type.h"
#include "common_tools.h"
#include <vector>

namespace kernel_selector {
namespace Tensor {

DataTensor::DataChannelArray DataTensor::dataChannelArray {{
    // explanation:
    // 0, 1, 2, 3, 4, 5, 6 means the ordering starts from X, then Y, then Z, then W, then F, then B
    // -1 means it's not used
    //                                     X,  Y,  Z,  W,  F,  B
    { DataLayout::f,                    { -1, -1, -1, -1,  0, -1 } },
    { DataLayout::bf,                   { -1, -1, -1, -1,  0,  1 } },
    { DataLayout::fb,                   { -1, -1, -1, -1,  1,  0 } },
    { DataLayout::bfyx,                 {  0,  1, -1, -1,  2,  3 } },
    { DataLayout::yxfb,                 {  2,  3, -1, -1,  1,  0 } },
    { DataLayout::byxf,                 {  1,  2, -1, -1,  0,  3 } },
    { DataLayout::fyxb,                 {  1,  2, -1, -1,  3,  0 } },
    { DataLayout::bfyx_f16,             {  0,  1, -1, -1,  2,  3 } },
    { DataLayout::b_fs_yx_fsv32,        {  0,  1, -1, -1,  2,  3 } },
    { DataLayout::b_fs_zyx_fsv32,       {  0,  1,  2, -1,  3,  4 } },
    { DataLayout::bs_f_bsv8__af8,       { -1, -1, -1, -1,  0,  1 } },
    { DataLayout::bs_f_bsv16__af8,      { -1, -1, -1, -1,  0,  1 } },
    { DataLayout::bf8_xy16,             {  0,  1, -1, -1,  2,  3 } },
    { DataLayout::winograd_2x3_s1_data, {  2,  1, -1, -1,  0,  3 } },
    { DataLayout::byxf_af32,            {  1,  2, -1, -1,  0,  3 } },
    { DataLayout::byx8_f4,              {  1,  2, -1, -1,  0,  3 } },
    { DataLayout::fs_bs_yx_bsv4_fsv32,  {  0,  1, -1, -1,  3,  2 } },
    { DataLayout::b_fs_yx_fsv4,         {  0,  1, -1, -1,  2,  3 } },
    { DataLayout::bfzyx,                {  0,  1,  2, -1,  3,  4 } },
    { DataLayout::fs_b_yx_fsv32,        {  0,  1, -1, -1,  3,  2 } },
    { DataLayout::b_fs_yx_32fp,         {  0,  1, -1, -1,  2,  3 } },
    { DataLayout::bfwzyx,               {  0,  1,  2,  3,  4,  5 } },
    { DataLayout::bfzyx_f16,            {  0,  1,  2, -1,  3,  4 } },
    { DataLayout::bfzyx_b16f16,         {  0,  1,  2, -1,  3,  4 } },
    { DataLayout::nv12,                 {  0,  1, -1, -1,  2,  3 } },
}};

WeightsTensor::WeightsChannelArray WeightsTensor::weightsChannelArray {{
    //                                                               X,  Y,  Z, IFM, OFM, LX, LY
    { WeightsLayout::oi,                                          { -1, -1, -1,   0,   1, -1, -1 } },
    { WeightsLayout::io,                                          { -1, -1, -1,   1,   0, -1, -1 } },
    { WeightsLayout::oiyx,                                        {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::oyxi,                                        {  1,  2, -1,   0,   3, -1, -1 } },
    { WeightsLayout::iyxo,                                        {  1,  2, -1,   3,   0, -1, -1 } },
    { WeightsLayout::yxio,                                        {  2,  3, -1,   1,   0, -1, -1 } },
    { WeightsLayout::oiyx_o16,                                    {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_iyx_osv16,                                {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_iyx_osv32,                                {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_iyx_osv64,                                {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::o_i_yx_i16_o16,                              {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_iyx_osv16_rotate_180,                     {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_i_osv8__ai8,                              { -1, -1, -1,   0,   1, -1, -1 } },
    { WeightsLayout::os_i_osv16__ai8,                             { -1, -1, -1,   0,   1, -1, -1 } },
    { WeightsLayout::os_i_osv16,                                  { -1, -1, -1,   0,   1, -1, -1 } },
    { WeightsLayout::i_yxs_os_yxsv2_osv16,                        {  1,  2, -1,   3,   0, -1, -1 } },
    { WeightsLayout::iy_xs_os_xsv2_osv16__ao32,                   {  1,  2, -1,   3,   0, -1, -1 } },
    { WeightsLayout::iy_xs_os_xsv2_osv8__ao32,                    {  1,  2, -1,   3,   0, -1, -1 } },
    { WeightsLayout::image_2d_weights_c4_fyx_b,                   {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::image_2d_weights_c1_b_fyx,                   {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::winograd_2x3_s1_weights,                     {  3,  2, -1,   1,   0, -1, -1 } },
    { WeightsLayout::winograd_2x3_s1_fused_weights,               {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::winograd_6x3_s1_fused_weights,               {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb,      {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb,      {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::dlstm_dir_io,                                {  1,  0, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_is_yx_isa8_osv8_isv4,                     {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_is_yx_isa8_osv8_isv4_swizzled_by_4,       {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_is_zyx_isa8_osv8_isv4,                    {  0,  1,  2,   3,   4, -1, -1 } },
    { WeightsLayout::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4,  {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4, {  0,  1,  2,   3,   4, -1, -1 } },
    { WeightsLayout::is_o_yx_isv32,                               {  1,  2, -1,   0,   3, -1, -1 } },
    { WeightsLayout::is_o32_yx_isv32_swizzled_by_4,               {  1,  2, -1,   0,   3, -1, -1 } },
    { WeightsLayout::os_is_y_x8_osv8_isv4,                        {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_is_y_x8_osv8_isv4_swizzled_by_4,          {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::bf_lyx_yx,                                   {  0,  1, -1,   4,   5,  2,  3 } },
    { WeightsLayout::os_is_yx_osv16_isv4,                         {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::os_is_yx_osv32_isv4_swizzled_by_2,           {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::oizyx,                                       {  0,  1,  2,   3,   4, -1, -1 } },
    { WeightsLayout::os_is_yx_osv32_isv32p,                       {  0,  1, -1,   2,   3, -1, -1 } },
    { WeightsLayout::o_i_zyx_i16_o16,                             {  0,  1,  2,   3,   4, -1, -1 } },
    { WeightsLayout::i_o_zyx_o16_i16,                             {  0,  1,  2,   4,   3, -1, -1 } },
    { WeightsLayout::os_is_osv32_isv32_swizzled_by_4,             { -1, -1, -1,   0,   1, -1, -1 } },
    { WeightsLayout::o_i_zyx_i8_o16_i2,                           {  0,  1,  2,   3,   4, -1, -1 } },
    { WeightsLayout::ozyxi_o16,                                   {  1,  2,  3,   0,   4, -1, -1 } },
}};

NDims DataTensor::GetSimpleDims(const std::vector<size_t>& d, DataLayout l) {
    std::vector<size_t> newDims = d;

    // TOOD: it's not the right pitches. it's here in order to calculate physical size
    switch (l) {
        case bs_f_bsv8__af8:
            assert(newDims.size() == 2);
            newDims[0] = RoundUp(newDims[0], 8);
            newDims[1] = RoundUp(newDims[1], 8);
            break;
        case bs_f_bsv16__af8:
            assert(newDims.size() == 2);
            newDims[0] = RoundUp(newDims[0], 8);
            newDims[1] = RoundUp(newDims[1], 16);
            break;
        case bfyx_f16:
            assert(newDims.size() == 4);
            newDims[2] = RoundUp(newDims[2], 16);
            break;
        case b_fs_yx_fsv32:
            assert(newDims.size() == 4);
            newDims[2] = RoundUp(newDims[2], 32);
            break;
        case b_fs_zyx_fsv32:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[3], 32);
            break;
        case bf8_xy16:
            assert(newDims.size() == 4);
            newDims[1] = RoundUp(newDims[1], 8);
            newDims[3] = RoundUp(newDims[2] * newDims[3], 16);
            newDims[2] = 1;
            break;
        case byxf_af32:
            assert(newDims.size() == 4);
            newDims[0] = RoundUp(newDims[0], 32);
            break;
        case byx8_f4:
            assert(newDims.size() == 4);
            newDims[0] = RoundUp(newDims[0], 4);
            newDims[1] = RoundUp(newDims[1], 8);
            break;
        case fs_bs_yx_bsv4_fsv32:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 32);
            newDims[2] = RoundUp(newDims[2], 4);
            break;
        case b_fs_yx_32fp:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 32);
            break;
        case fs_b_yx_fsv32:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 32);
            break;
        case bfzyx_f16:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[3], 16);
            break;
        case bfzyx_b16f16:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[3], 16);
            newDims[4] = RoundUp(newDims[4], 16);
            break;
        default:
            break;
    }

    NDims ret(newDims.size());
    size_t pitch = 1;

    for (size_t i = 0; i < newDims.size(); i++) {
        Pad p = {0, newDims[i] - d[i]};
        ret[i] = {d[i], pitch, p};
        pitch *= newDims[i];
    }

    if (l == byxf_af32 || l == fs_bs_yx_bsv4_fsv32 || l == byx8_f4) {
        ret[0].pitch = 1;
        ret[1].pitch = ret[0].pitch * newDims[0];
        ret[2].pitch = ret[1].pitch * newDims[1];
        ret[3].pitch = ret[2].pitch * newDims[2];
        ret[4].pitch = ret[3].pitch * newDims[3];
    }

    return ret;
}

DataTensor DataTensor::TransformIgnorePadding(DataLayout l) const {
    const uint32_t src_channels = ChannelsCount(layout);
    const uint32_t dst_channels = ChannelsCount(l);

    const size_t src_x = X().v;
    const size_t src_y = Y().v;
    const size_t src_z = Z().v;
    const size_t src_w = W().v;

    std::vector<size_t> vec(dst_channels);
    if (src_channels == 2 && dst_channels == 2) {
        vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 4 && dst_channels == 4) {
        vec[Channelndex(l, DataChannelName::X)] = X().v;
        vec[Channelndex(l, DataChannelName::Y)] = Y().v;
        vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 2 && dst_channels == 4) {
        const size_t dst_ifm = Feature().v / (src_x * src_y);
        const size_t dst_xy = Feature().v % (src_x * src_y);
        const size_t dst_y = dst_xy / src_x;
        const size_t dst_x = dst_xy % src_x;
        vec[Channelndex(l, DataChannelName::X)] = dst_x;
        vec[Channelndex(l, DataChannelName::Y)] = dst_y;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 4 && dst_channels == 2) {
        const size_t dst_ifm = Feature().v * src_x * src_y;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 2 && dst_channels == 5) {
        const size_t dst_ifm = Feature().v / (src_x * src_y * src_z);
        const size_t dst_xyz = Feature().v % (src_x * src_y * src_z);
        const size_t dst_x = dst_xyz % src_x;
        const size_t dst_yz = dst_xyz / src_x;
        const size_t dst_y = dst_yz % src_y;
        const size_t dst_z = dst_yz / src_y;

        vec[Channelndex(l, DataChannelName::X)] = dst_x;
        vec[Channelndex(l, DataChannelName::Y)] = dst_y;
        vec[Channelndex(l, DataChannelName::Z)] = dst_z;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 5 && dst_channels == 2) {
        const size_t dst_ifm = Feature().v * src_x * src_y * src_z;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 5 && dst_channels == 5) {
        vec[Channelndex(l, DataChannelName::X)] = X().v;
        vec[Channelndex(l, DataChannelName::Y)] = Y().v;
        vec[Channelndex(l, DataChannelName::Z)] = Z().v;
        vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 6 && dst_channels == 6) {
        vec[Channelndex(l, DataChannelName::X)] = X().v;
        vec[Channelndex(l, DataChannelName::Y)] = Y().v;
        vec[Channelndex(l, DataChannelName::Z)] = Z().v;
        vec[Channelndex(l, DataChannelName::W)] = W().v;
        vec[Channelndex(l, DataChannelName::FEATURE)] = Feature().v;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 6 && dst_channels == 2) {
        const size_t dst_ifm = Feature().v * src_x * src_y * src_z * src_w;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 2 && dst_channels == 6) {
        const size_t dst_ifm = Feature().v / (src_x * src_y * src_z * src_w);
        const size_t dst_xyzw = Feature().v % (src_x * src_y * src_z * src_w);
        const size_t dst_x = dst_xyzw % src_x;
        const size_t dst_yzw = dst_xyzw / src_x;
        const size_t dst_y = dst_yzw % src_y;
        const size_t dst_zw = dst_yzw / src_y;
        const size_t dst_z = dst_zw % src_z;
        const size_t dst_w = dst_zw / src_z;

        vec[Channelndex(l, DataChannelName::X)] = dst_x;
        vec[Channelndex(l, DataChannelName::Y)] = dst_y;
        vec[Channelndex(l, DataChannelName::Z)] = dst_z;
        vec[Channelndex(l, DataChannelName::W)] = dst_w;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst_ifm;
        vec[Channelndex(l, DataChannelName::BATCH)] = Batch().v;
    } else if (src_channels == 2 && dst_channels == 1) {
        const size_t dst = Feature().v * Batch().v;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst;
    } else if (src_channels == 4 && dst_channels == 1) {
        const size_t dst = Feature().v * Batch().v * src_x * src_y;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst;
    } else if (src_channels == 5 && dst_channels == 1) {
        const size_t dst = Feature().v * Batch().v * src_x * src_y * src_z;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst;
    } else if (src_channels == 6 && dst_channels == 1) {
        const size_t dst = Feature().v * Batch().v * src_x * src_y * src_z * src_w;
        vec[Channelndex(l, DataChannelName::FEATURE)] = dst;

    } else {
        assert(0);
    }

    return {vec, dtype, l};
}

DataTensor DataTensor::FlattenFeatureAndSpatials() const {
    DataLayout l;

    const auto x = X();
    const auto y = Y();
    const auto z = Z();
    const auto w = W();
    const auto f = Feature();
    const auto b = Batch();

    DataLayout targetLayout = Tensor::bf;
    switch (layout) {
        case Tensor::bf:
        case Tensor::fb:
            return *this;

        case Tensor::fyxb:
            targetLayout = Tensor::fb;

            // TODO: [FUTURE] Use C++17 [[fallthrough]] instead of code duplication to get portable warning avoidance.
            if (f.pitch == y.v * x.v * x.pitch) {  // no padding in X/Y axis
                l = targetLayout;
                break;
            }
            throw std::runtime_error("Unsupported - cannot flatten with padding");

        case Tensor::bfyx:
            if (f.pitch == y.v * x.v * x.pitch) {  // no padding in X/Y axis
                l = targetLayout;
                break;
            }
            throw std::runtime_error("Unsupported - cannot flatten with padding");

        case Tensor::bfzyx:
            if (f.pitch == z.v * y.v * x.v * x.pitch) {  // no padding in X/Y/Z axis
                l = targetLayout;
                break;
            }
            throw std::runtime_error("Unsupported - cannot flatten with padding");
        case Tensor::bfwzyx:
            if (f.pitch == w.v * z.v * x.v * y.v * x.pitch) {
                l = targetLayout;
                break;
            }
            throw std::runtime_error("Unsupported - cannot flatten with padding");
        case Tensor::yxfb:
            targetLayout = Tensor::fb;

            // TODO: [FUTURE] Use C++17 [[fallthrough]] instead of code duplication to get portable warning avoidance.
            if ((x.pitch == f.pitch && y.pitch == x.v * x.pitch) ||                                // YX - no Features (val/pitch)
                (y.v == 1 && x.v == 1 && x.pitch == f.pitch && y.pitch == f.pitch) ||              // Feature only
                (f.v * f.pitch == x.pitch && f.v * f.pitch == y.pitch && y.v == 1 && x.v == 1)) {  // Feature only
                l = targetLayout;
                break;
            }
            throw std::runtime_error("Unsupported - cannot flatten yxf to f if f/yx != 1");

        case Tensor::byxf:
            if ((x.pitch == f.pitch && y.pitch == x.v * x.pitch) ||                               // YX - no Features (val/pitch)
                (y.v == 1 && x.v == 1 && x.pitch == f.pitch && y.pitch == f.pitch) ||             // Feature only
                (f.v * f.pitch == x.pitch && f.v * f.pitch == y.pitch && y.v == 1 && x.v == 1)) {  // Feature only
                l = targetLayout;
                break;
            }
            throw std::runtime_error("Unsupported - cannot flatten yxf to f if f/yx != 1");
        default:
            throw std::runtime_error("Unsupported - unsupported layout");
            break;
    }

    DataTensor res = TransformIgnorePadding(l);

    if (l == DataLayout::bf) {
        res.dims[Channelndex(l, DataChannelName::BATCH)].pitch = b.pitch;
        res.dims[Channelndex(l, DataChannelName::BATCH)].pad = b.pad;
    } else {
        res.dims[Channelndex(l, DataChannelName::FEATURE)].pitch = dims[Channelndex(l, DataChannelName::BATCH) + 1].pitch;
        res.dims[Channelndex(l, DataChannelName::FEATURE)].pad = dims[Channelndex(l, DataChannelName::BATCH) + 1].pad;
    }

    return res;
}

DataTensor DataTensor::FlattenEverything() const {
    DataLayout targetLayout = Tensor::f;
    DataTensor res = TransformIgnorePadding(targetLayout);
    return res;
}

NDims WeightsTensor::GetSimpleDims(const std::vector<size_t>& d, WeightsLayout l) {
    std::vector<size_t> newDims = d;

    // TOOD: it's not the right pitches. it's here in order to calculate physical size
    switch (l) {
        case os_iyx_osv16:
        case os_iyx_osv16_rotate_180:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 16);
            break;
        case os_iyx_osv32:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 32);
            break;
        case os_iyx_osv64:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 64);
            break;
        case os_i_osv8__ai8:
            assert(newDims.size() == 2);
            newDims[0] = RoundUp(newDims[0], 8);
            newDims[1] = RoundUp(newDims[1], 8);
            break;
        case os_i_osv16__ai8:
            assert(newDims.size() == 2);
            newDims[0] = RoundUp(newDims[0], 8);
            newDims[1] = RoundUp(newDims[1], 16);
            break;
        case os_i_osv16:
            assert(newDims.size() == 2);
            newDims[1] = RoundUp(newDims[1], 16);
            break;
        case os_is_osv32_isv32_swizzled_by_4:
            assert(newDims.size() == 2);
            newDims[0] = RoundUp(newDims[0], 32);
            newDims[1] = RoundUp(newDims[1], 32);
            break;
        case i_yxs_os_yxsv2_osv16:
            assert(newDims.size() == 4);
            newDims[0] = RoundUp(newDims[0], 16);
            break;
        case iy_xs_os_xsv2_osv16__ao32:
        case iy_xs_os_xsv2_osv8__ao32:
            assert(newDims.size() == 4);
            newDims[0] = RoundUp(newDims[0], 32);
            break;
        case os_is_yx_isa8_osv8_isv4:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 8);
            newDims[2] = RoundUp(newDims[2], 32);
            break;
        case os_is_zyx_isa8_osv8_isv4:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[3], 32);
            newDims[4] = RoundUp(newDims[4], 8);
            break;
        case os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 32);
            newDims[2] = RoundUp(newDims[2], 32);
            break;
        case os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4:
            assert(newDims.size() == 5);
            newDims[4] = RoundUp(newDims[4], 32);
            newDims[3] = RoundUp(newDims[3], 32);
            break;
        case os_is_yx_isa8_osv8_isv4_swizzled_by_4:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 32);
            newDims[2] = RoundUp(newDims[2], 32);
            break;
        case is_o_yx_isv32:
            assert(newDims.size() == 4);
            newDims[0] = RoundUp(newDims[0], 32);
            break;
        case is_o32_yx_isv32_swizzled_by_4:
            assert(newDims.size() == 4);
            newDims[0] = RoundUp(newDims[0], 32);
            newDims[3] = RoundUp(newDims[3], 32);
            break;
        case os_is_y_x8_osv8_isv4:
        case os_is_y_x8_osv8_isv4_swizzled_by_4:
            assert(newDims.size() == 4);
            newDims[2] = RoundUp(newDims[2], 4);
            newDims[3] = RoundUp(newDims[3], 8);
            newDims[0] = RoundUp(newDims[0], 8);
            break;
        case os_is_yx_osv16_isv4:
            assert(newDims.size() == 4);
            newDims[2] = RoundUp(newDims[2], 4);
            newDims[3] = RoundUp(newDims[3], 16);
            break;
        case os_is_yx_osv32_isv4_swizzled_by_2:
            assert(newDims.size() == 4);
            newDims[2] = RoundUp(newDims[2], 4);
            newDims[3] = RoundUp(newDims[3], 32);
            break;
        case os_is_yx_osv32_isv32p:
            assert(newDims.size() == 4);
            newDims[2] = RoundUp(newDims[2], 32);  // ic
            newDims[3] = RoundUp(newDims[3], 32);  // oc
            break;
        case o_i_yx_i16_o16:
            assert(newDims.size() == 4);
            newDims[2] = RoundUp(newDims[2], 16);
            newDims[3] = RoundUp(newDims[3], 16);
            break;
        case oiyx_o16:
            assert(newDims.size() == 4);
            newDims[3] = RoundUp(newDims[3], 16);
            break;
        case o_i_zyx_i16_o16:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[3], 16);
            newDims[4] = RoundUp(newDims[4], 16);
            break;
        case i_o_zyx_o16_i16:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[3], 16);
            newDims[4] = RoundUp(newDims[4], 16);
            break;
        case o_i_zyx_i8_o16_i2:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[3], 16);
            newDims[4] = RoundUp(newDims[4], 16);
            break;
        case ozyxi_o16:
            assert(newDims.size() == 5);
            newDims[3] = RoundUp(newDims[0], 16);
            break;
        default:
            break;
    }

    NDims ret(newDims.size());
    size_t pitch = 1;

    for (size_t i = 0; i < newDims.size(); i++) {
        Pad p = {0, newDims[i] - d[i]};
        ret[i] = {d[i], pitch, p};
        pitch *= newDims[i];
    }

    if (l == i_yxs_os_yxsv2_osv16) {
        ret[3].pitch = RoundUp(ret[1].v * ret[2].v, 2) * ret[1].pitch;
        ret[2].pad.after = newDims[2] - ret[2].v;
    } else if (l == iy_xs_os_xsv2_osv16__ao32 ||
               l == iy_xs_os_xsv2_osv8__ao32) {
        ret[2].pitch = RoundUp(ret[1].v, 2) * ret[1].pitch;
        ret[1].pad.after = newDims[1] - ret[1].v;

        ret[3].pitch = ret[2].v * ret[2].pitch;
        ret[2].pad.after = newDims[2] - ret[2].v;
    } else if (l == os_is_yx_isa8_osv8_isv4 || l == os_is_yx_isa8_osv8_isv4_swizzled_by_4) {
        ret[0].pitch = 256;
        ret[1].pitch = ret[0].pitch * ret[0].v;
    }

    return ret;
}

WeightsTensor WeightsTensor::TransformIgnorePadding(WeightsLayout l, WeightsType t) const {
    const uint32_t src_channels = ChannelsCount(layout);
    const uint32_t dst_channels = ChannelsCount(l);

    const size_t src_x = X().v;
    const size_t src_y = Y().v;
    const size_t src_z = Z().v;

    std::vector<size_t> vec(dst_channels);
    if (src_channels == 2 && dst_channels == 2) {
        vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
    } else if (src_channels == 4 && dst_channels == 4) {
        vec[Channelndex(l, WeightsChannelName::X)] = X().v;
        vec[Channelndex(l, WeightsChannelName::Y)] = Y().v;
        vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;

        // requirement for winograd 2x3
        if (l == WeightsLayout::winograd_2x3_s1_weights || l == WeightsLayout::winograd_2x3_s1_fused_weights) {
            vec[Channelndex(l, WeightsChannelName::X)] = 4;
            vec[Channelndex(l, WeightsChannelName::Y)] = 3;
        } else if (l == WeightsLayout::winograd_6x3_s1_fused_weights) {
            vec[Channelndex(l, WeightsChannelName::X)] = 8;
            vec[Channelndex(l, WeightsChannelName::Y)] = 3;
        }
    } else if (src_channels == 2 && dst_channels == 4) {
        const size_t dst_ifm = IFM().v / (src_x * src_y);
        const size_t dst_xy = IFM().v % (src_x * src_y);
        const size_t dst_y = dst_xy / src_x;
        const size_t dst_x = dst_xy % src_x;
        vec[Channelndex(l, WeightsChannelName::X)] = dst_x;
        vec[Channelndex(l, WeightsChannelName::Y)] = dst_y;
        vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
    } else if (src_channels == 4 && dst_channels == 2) {
        const size_t dst_ifm = IFM().v * src_x * src_y;
        vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
    } else if (src_channels == 2 && dst_channels == 5) {
        const size_t dst_ifm = IFM().v / (src_x * src_y * src_z);
        const size_t dst_xyz = IFM().v % (src_x * src_y * src_z);
        const size_t dst_x = dst_xyz % src_x;
        const size_t dst_yz = dst_xyz / src_x;
        const size_t dst_y = dst_yz % src_y;
        const size_t dst_z = dst_yz / src_y;
        vec[Channelndex(l, WeightsChannelName::X)] = dst_x;
        vec[Channelndex(l, WeightsChannelName::Y)] = dst_y;
        vec[Channelndex(l, WeightsChannelName::Z)] = dst_z;
        vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
    } else if (src_channels == 5 && dst_channels == 2) {
        const size_t dst_ifm = IFM().v * src_x * src_y * src_z;
        vec[Channelndex(l, WeightsChannelName::IFM)] = dst_ifm;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
    } else if (src_channels == 5 && dst_channels == 5) {
        vec[Channelndex(l, WeightsChannelName::X)] = X().v;
        vec[Channelndex(l, WeightsChannelName::Y)] = Y().v;
        vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
        vec[Channelndex(l, WeightsChannelName::Z)] = Z().v;
    } else if (src_channels == 6 && dst_channels == 6) {
        vec[Channelndex(l, WeightsChannelName::X)] = X().v;
        vec[Channelndex(l, WeightsChannelName::Y)] = Y().v;
        vec[Channelndex(l, WeightsChannelName::IFM)] = IFM().v;
        vec[Channelndex(l, WeightsChannelName::OFM)] = OFM().v;
        vec[Channelndex(l, WeightsChannelName::LX)] = LX().v;
        vec[Channelndex(l, WeightsChannelName::LY)] = LY().v;
    } else {
        assert(0);
    }

    return {vec, t, l};
}
}  // namespace Tensor
}  // namespace kernel_selector
