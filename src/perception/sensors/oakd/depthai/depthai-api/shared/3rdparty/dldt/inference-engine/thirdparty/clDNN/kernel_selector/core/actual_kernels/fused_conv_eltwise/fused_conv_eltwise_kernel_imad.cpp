﻿// Copyright (c) 2019 Intel Corporation
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


#include "fused_conv_eltwise_kernel_imad.h"
#include "common_tools.h"
#include "kernel_selector_utils.h"
#include <vector>

//
// Kernel specific constants
//
#define SIMD_SIZE 16

static bool getOutBlock_WH(size_t output_size,
                           size_t stride,
                           size_t kernel_size,
                           size_t& output_block_w,
                           size_t& output_block_h) {
    bool verify_output_ranges = false;

    output_block_w = output_block_h = 0;

    size_t upper_border = output_size < SIMD_SIZE ? output_size : SIMD_SIZE;

    size_t stride_restrictions = (SIMD_SIZE - (kernel_size - 1)) / stride;

    size_t max_posible_tile_size = upper_border < stride_restrictions ? upper_border : stride_restrictions;

    if (output_size % max_posible_tile_size == 0) {
        output_block_w = max_posible_tile_size;
    } else {
        size_t min_horisontal_block_size = 2;  // 4;

        size_t block_size = 0;

        for (size_t i = min_horisontal_block_size; i < max_posible_tile_size; i++) {
            if (output_size % i == 0)
                block_size = i;
        }

        if (block_size != 0) {
            output_block_w = block_size;
        } else {
            output_block_w = max_posible_tile_size;
            verify_output_ranges = true;
        }
    }

    if (output_block_w <= 4)
        output_block_h = output_block_w;
    else
        output_block_h = 1;

    return verify_output_ranges;
}
namespace kernel_selector {

ParamsKey fused_conv_eltwise_kernel_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::byxf_af32);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableFusedConvEltwInt8Quantization();
    k.EnableFusedConvEltwOutputCalibration();
    k.DisableTuning();
    k.EnableFusedConvEltwiseRWOutOpt();
    k.EnableEltwiseStride();
    return k;
}

KernelsData fused_conv_eltwise_kernel_imad::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants fused_conv_eltwise_kernel_imad::GetJitConstants(const fused_conv_eltwise_params& params,
                                                             const DispatchData& kd) const {
    auto mem_consts = Parent::GetJitConstants(params, kd);

    const auto& input = params.inputs[0];
    const auto& output = params.output;
    mem_consts.Merge(MakeActivationJitConstants(params.conv.activations, GetUnitType(params), "_CONV_TYPED", true));
    mem_consts.Merge(MakeActivationJitConstants(params.activations, GetUnitType(params), "_ELTW_TYPED", true));
    mem_consts.Merge(MakeTypeJitConstants(Datatype::F32, "float"));

    const auto& iDims = input.GetDims();
    const auto& oDims = output.GetDims();
    const auto& weights = params.weights;
    const auto& wDims = weights.GetDims();
    const int iX = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::X);
    const int iY = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::Y);
    const int iF = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::FEATURE);
    const int wOD = WeightsTensor::Channelndex(weights.GetLayout(), Tensor::WeightsChannelName::OFM);
    const int oX = DataTensor::Channelndex(output.GetLayout(), Tensor::DataChannelName::X);
    const int oY = DataTensor::Channelndex(output.GetLayout(), Tensor::DataChannelName::Y);
    mem_consts.AddConstants({
        MakeJitConstant("_IW", iDims[iX].v),
        MakeJitConstant("_IH", iDims[iY].v),
        MakeJitConstant("_ID", RoundUp(iDims[iF].v, 4)),
        MakeJitConstant("IWPAD", iDims[iX].pad.before + iDims[iX].pad.after),
        MakeJitConstant("IHPAD", iDims[iY].pad.before + iDims[iY].pad.after),
        MakeJitConstant("_OW", oDims[oX].v),
        MakeJitConstant("_OH", oDims[oY].v),
        MakeJitConstant("_OD", wDims[wOD].v),
        MakeJitConstant("OWPAD", oDims[oX].pad.before + oDims[oX].pad.after),
        MakeJitConstant("OHPAD", oDims[oY].pad.before + oDims[oY].pad.after),
        MakeJitConstant("SIMD_SIZE", SIMD_SIZE),
        MakeJitConstant("K_HEIGHT", wDims[iY].v),
        MakeJitConstant("K_WIDTH", wDims[iX].v),
        MakeJitConstant("K_STRIDE", params.conv.stride.x),  // X and Y must be equal
        MakeJitConstant("NON_BLOCK_LOAD", 1),
    });

    size_t obw, obh;
    bool verify_output_ranges = getOutBlock_WH(oDims[oX].v, params.conv.stride.x, wDims[iX].v, obw, obh);
    mem_consts.AddConstants({MakeJitConstant("OUT_BLOCK_WIDTH", obw),
                             MakeJitConstant("OUT_BLOCK_HEIGHT", obh),
                             MakeJitConstant("NEED_TO_VERIFY_OUTPUT_RANGES", verify_output_ranges)});
    if (params.non_conv_scale != 1.0f)
        mem_consts.AddConstant(MakeJitConstant("NON_CONV_SCALE", params.non_conv_scale));

    return mem_consts;
}  // GetJitConstants

fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_imad::SetDefault(
    const fused_conv_eltwise_params& params,
    int) const {
    DispatchData kd;

    const auto& in = params.inputs[0];
    const auto& weights = params.weights;
    const auto& iDims = in.GetDims();
    const auto& wDims = weights.GetDims();
    const int iX = DataTensor::Channelndex(in.GetLayout(), Tensor::DataChannelName::X);
    const int iY = DataTensor::Channelndex(in.GetLayout(), Tensor::DataChannelName::Y);
    const int iB = DataTensor::Channelndex(in.GetLayout(), Tensor::DataChannelName::BATCH);
    const int wOD = WeightsTensor::Channelndex(weights.GetLayout(), Tensor::WeightsChannelName::OFM);

    size_t otw, oth;
    getOutBlock_WH(iDims[iX].v, params.conv.stride.x, iDims[iX].pad.before + iDims[iX].pad.after, otw, oth);

    size_t dim_add = ((wDims[wOD].v * iDims[iB].v) % SIMD_SIZE);
    if (dim_add != 0)
        dim_add = SIMD_SIZE - dim_add;


    std::vector<size_t> global = {// globalRange[0] = ((_IW / K_STRIDE) + (OTW - 1)) / OTW;
                                  // number of tiles needed to cover output width
                                  (((iDims[iX].v / params.conv.stride.x) + (otw - 1)) / otw),

                                  // globalRange[1] = ((_IH / K_STRIDE) + (OTH - 1)) / OTH;
                                  // number of tiles needed to cover output height
                                  (((iDims[iY].v / params.conv.stride.y) + (oth - 1)) / oth),

                                  // globalRange[2] = (_OD * _B) + ((_B *_OD) % __WORKGROUP_SIZE);
                                  // round depth range up
                                  ((wDims[wOD].v * iDims[iB].v) + dim_add)};

    std::vector<size_t> local = {1, 1, SIMD_SIZE};

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    kd.cldnnStyle = {0, 0, 0, 0, 0};
    kd.gemmStyle = {0, 0, 0, 0, 0, 0};
    kd.effiency = FORCE_PRIORITY_2;

    return kd;
}  // SetDefault

bool fused_conv_eltwise_kernel_imad::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    KernelData kd = KernelData::Default<fused_conv_eltwise_params>(params);
    fused_conv_eltwise_params& newParams = *static_cast<fused_conv_eltwise_params*>(kd.params.get());

    if (newParams.conv.stride.x != newParams.conv.stride.y) {
        // Strides must be equial
        return false;
    } else if ((newParams.conv.filterSize.x != m_FilterSizeX) || (newParams.conv.filterSize.y != m_FilterSizeY)) {
        // Kernel does not support such filter size
        return false;
    }

    return true;
}
}  // namespace kernel_selector
