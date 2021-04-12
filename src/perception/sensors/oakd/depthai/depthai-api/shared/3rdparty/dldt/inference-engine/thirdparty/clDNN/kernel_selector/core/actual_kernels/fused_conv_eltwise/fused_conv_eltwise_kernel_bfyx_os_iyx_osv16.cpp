﻿// Copyright (c) 2018 Intel Corporation
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


#include "fused_conv_eltwise_kernel_bfyx_os_iyx_osv16.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace kernel_selector {
// Sub-group size used by "kernel_name_bfyx_os_iyx_osv16" kernel.
constexpr size_t sub_group_size = 16;

fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::fused_conv_eltwise_kernel_bfyx_os_iyx_osv16()
    : fused_conv_eltwise_kernel_base("fused_conv_eltwise_gpu_bfyx_os_iyx_osv16") {
    // Generate the dispatch options to the auto-tuner.
    std::vector<size_t> blockWidthSizes = {1, 2, 4, 5, 6, 8, 10, 12, 14, 16};
    std::vector<size_t> blockHeightSizes = {1, 2, 3, 4, 5};
    std::vector<size_t> prefetchSizes = {1, 2, 3, 4, 5, 6, 8, 10};
    std::vector<std::string> executionModes = fused_conv_eltwise_kernel_base::autoTuneOptions;
    const size_t maxBlockSize = 60;

    for (auto executionMode : executionModes) {
        for (auto blockWidth : blockWidthSizes) {
            for (auto blockHeight : blockHeightSizes) {
                for (auto prefetch : prefetchSizes) {
                    if (blockWidth * blockHeight <= maxBlockSize) {
                        autoTuneOptions.emplace_back(AutoTuneOption{blockWidth, blockHeight, prefetch, executionMode});
                    }
                }
            }
        }
    }
}

ParamsKey fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableFusedConvEltwSplitSupport();
    k.EnableFusedConvEltwDilation();
    k.EnableFusedConvEltwTranspose();
    k.EnableFusedConvEltwiseRWOutOpt();  // data for second input are already in output
    return k;
}

static std::pair<size_t, size_t> get_bfyx_req_input_block_dims(size_t output_block_width,
                                                               size_t output_block_height,
                                                               const uSize& filter_size,
                                                               const uSize& stride,
                                                               const uSize& dilation,
                                                               size_t sg_size = 16,
                                                               size_t read_chunk_size = 8,
                                                               size_t min_read_size = 16) {
    assert(output_block_width > 0 && output_block_height > 0);
    assert(stride.x > 0 && stride.y > 0);
    assert(filter_size.x > 0 && filter_size.y > 0);

    // Number of elements in X dimension needed from input to compute output block without re-reading input.
    size_t input_block_req_width = (output_block_width - 1) * stride.x + (filter_size.x - 1) * dilation.x + 1;
    // Number of elements in Y dimension needed from input to compute output block without re-reading input.
    size_t input_block_req_height = (output_block_height - 1) * stride.y + (filter_size.y - 1) * dilation.y + 1;

    // Required number of elements in X dimension rounded to nearest >= read chunk size.
    size_t input_block_read_width = std::max(RoundUp(input_block_req_width, read_chunk_size), min_read_size);
    // Number of sub-group-sized vectors of unit type needed to store input block.
    size_t input_block_array_size = CeilDiv(input_block_req_height * input_block_read_width, sg_size);

    return std::make_pair(input_block_array_size, input_block_read_width);
}

static void shrink_blocks_to_output_size(size_t output_x, size_t output_y, size_t& block_x, size_t& block_y) {
    // how many elements we will compute in each dimension
    size_t computed_x = Align(output_x, block_x);
    size_t computed_y = Align(output_y, block_y);
    // how many simds we need in each dimension
    size_t simds_x = computed_x / block_x;
    size_t simds_y = computed_y / block_y;
    // how many unused values we have in each dimension
    size_t unused_x = computed_x - output_x;
    size_t unused_y = computed_y - output_y;

    block_x -= unused_x / simds_x;
    block_y -= unused_y / simds_y;
}

fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::AutoTuneOption
fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::GetAutoTuneOptions(const Params& p, int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    AutoTuneOption option = {0, 0, 0, DEFAULT};

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    if (cp.stride.x == 1 && cp.stride.y == 1) {
        if (cp.filterSize.x == 1 && cp.filterSize.y == 1) {
            option.blockWidth = 16;
            option.blockHeight = 1;
            option.prefetch = 4;
        // if less than 16 values is required to compute one single row of output
        // then each WI shall compute one single row to maximize reuse within SIMD subgroup (this gives very nice
        // performance results)
        } else if (cp.output.X().v + (cp.filterSize.x - 1) * cp.dilation.x < sub_group_size) {
            option.blockWidth = cp.output.X().v;
            option.blockHeight = 1;
            option.prefetch = 4;
        } else if (cp.filterSize.x < 5 && cp.filterSize.y < 5) {
            option.blockWidth = sub_group_size - cp.filterSize.x + 1;
            option.blockHeight = 2;
            option.prefetch = 4;
        } else {
            option.blockWidth = 4;
            option.blockHeight = 3;
            option.prefetch = 4;
        }
    } else if (cp.stride.x == 2 && cp.stride.y == 2) {
        option.blockWidth = 5;
        option.blockHeight = 4;
        option.prefetch = 4;
    } else {
        option.blockWidth = 4;
        option.blockHeight = 3;
        option.prefetch = 5;
        // run_info.effiency = FORCE_PRIORITY_7; // GEMM is better
    }

    // if this is not 1x1 batch1 case then shrink filters, other way we're memory bound and it's best to use 16x1 block
    // sizes
    if (cp.filterSize.x != 1 || cp.filterSize.y != 1 || cp.output.Batch().v != 1) {
        shrink_blocks_to_output_size(cp.output.X().v, cp.output.Y().v, option.blockWidth, option.blockHeight);
    }

    return option;
}

fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::SetDefault(
    const fused_conv_eltwise_params& cp,
    int autoTuneIndex) const {
    DispatchData runInfo = fused_conv_eltwise_kernel_base::SetDefault(cp);

    const auto of_maps = cp.output.Feature().v;
    const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);

    runInfo.effiency = FORCE_PRIORITY_3;

    auto tuneOptions = GetAutoTuneOptions(cp, autoTuneIndex);
    runInfo.cldnnStyle.blockWidth = tuneOptions.blockWidth;
    runInfo.cldnnStyle.blockHeight = tuneOptions.blockHeight;
    runInfo.cldnnStyle.prefetch = tuneOptions.prefetch;

    auto input_block_dims = get_bfyx_req_input_block_dims(runInfo.cldnnStyle.blockWidth,
                                                          runInfo.cldnnStyle.blockHeight,
                                                          cp.conv.filterSize,
                                                          cp.conv.stride,
                                                          cp.conv.dilation,
                                                          sub_group_size,
                                                          runInfo.fp16UnitUsed ? sub_group_size : sub_group_size / 2,
                                                          sub_group_size);
    runInfo.cldnnStyle.inputBlockArraySize = input_block_dims.first;
    runInfo.cldnnStyle.inputBlockWidth = input_block_dims.second;

    runInfo.gws0 = CeilDiv(cp.output.X().v, runInfo.cldnnStyle.blockWidth);
    runInfo.gws1 = CeilDiv(cp.output.Y().v, runInfo.cldnnStyle.blockHeight);
    runInfo.gws2 = of_threads_per_batch * cp.output.Batch().v;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = sub_group_size;

    return runInfo;
}

bool fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::Validate(const Params& p, const optional_params& o) const {
    if (!fused_conv_eltwise_kernel_base::Validate(p, o) || !FusedConvolutionEltwiseCheckInput(p, o)) {
        return false;
    }

    return true;
}

JitConstants fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::GetJitConstants(const fused_conv_eltwise_params& params,
                                                                          const DispatchData& runInfo) const {
    const auto of_maps = params.output.Feature().v;
    const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);
    size_t leftovers = of_threads_per_batch - of_maps;

    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", runInfo.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", runInfo.cldnnStyle.blockHeight));
    jit.AddConstant(MakeJitConstant("IN_BLOCK_ARRAY_SIZE", runInfo.cldnnStyle.inputBlockArraySize));
    jit.AddConstant(MakeJitConstant("IN_BLOCK_WIDTH", runInfo.cldnnStyle.inputBlockWidth));
    jit.AddConstant(MakeJitConstant("PREFETCH", runInfo.cldnnStyle.prefetch));

    if (leftovers) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS", leftovers));
    }

    return jit;
}

std::vector<WeightsLayout> fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::GetSupportedWeightLayouts(
    const fused_conv_eltwise_params& params) const {
    if (!params.conv.transposed) {
        return {WeightsLayout::os_iyx_osv16};
    } else {
        return {WeightsLayout::os_iyx_osv16_rotate_180};
    }
}

KernelsData fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::GetKernelsData(const Params& params,
                                                                        const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData fused_conv_eltwise_kernel_bfyx_os_iyx_osv16::GetKernelsDataForAutoTune(
    const Params& params,
    const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelsData res = {};

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options, static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

}  // namespace kernel_selector
