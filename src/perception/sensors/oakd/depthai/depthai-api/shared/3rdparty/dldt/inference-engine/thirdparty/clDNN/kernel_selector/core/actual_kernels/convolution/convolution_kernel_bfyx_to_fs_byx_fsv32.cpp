// Copyright (c) 2019 Intel Corporation
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

#include "convolution_kernel_bfyx_to_fs_byx_fsv32.h"
#include <vector>

namespace kernel_selector {

static constexpr size_t subGroupSize = 16;
static constexpr size_t fsv = 32;
static constexpr size_t fsvPerThread = fsv / subGroupSize;
static constexpr size_t maxBlockSize = 48;

ConvolutionKernel_bfyx_to_fs_byx_fsv32::ConvolutionKernel_bfyx_to_fs_byx_fsv32()
    : ConvolutionKernelBase("convolution_gpu_bfyx_to_fs_byx_fsv32") {
    std::vector<size_t> blockWidths = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<size_t> blockHeights = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : blockWidths) {
        for (auto h : blockHeights) {
            if (w * h <= maxBlockSize) {
                for (auto exeMode : executionModes) {
                    autoTuneOptions.emplace_back(AutoTuneOption{w, h, exeMode});
                }
            }
        }
    }
}

ParamsKey ConvolutionKernel_bfyx_to_fs_byx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDilation();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

size_t getInputSize(size_t stride, size_t filterSize, size_t dilation, size_t blockWidth) {
    return (blockWidth - 1) * stride + (filterSize - 1) * dilation + 1;
}

ConvolutionKernel_bfyx_to_fs_byx_fsv32::AutoTuneOption ConvolutionKernel_bfyx_to_fs_byx_fsv32::GetAutoTuneOptions(
    const Params&,
    int autoTuneIndex) const {
    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(autoTuneOptions.size()))
        return autoTuneOptions[autoTuneIndex];

    return {8, 2, AGE_BASED};
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_to_fs_byx_fsv32::SetDefault(const convolution_params& arg,
                                                                                       int autoTuneIndex) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    AutoTuneOption option = GetAutoTuneOptions(arg, autoTuneIndex);

    runInfo.effiency = FORCE_PRIORITY_3;

    runInfo.cldnnStyle.blockHeight = option.blockHeight;
    runInfo.cldnnStyle.blockWidth = option.blockWidth;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = 16;

    runInfo.gws0 = CeilDiv(arg.output.X().v, option.blockWidth);
    runInfo.gws1 = CeilDiv(arg.output.Y().v, option.blockHeight);
    runInfo.gws2 = CeilDiv(arg.output.Feature().v, 32) * 16 * arg.output.Batch().v;

    return runInfo;
}

bool ConvolutionKernel_bfyx_to_fs_byx_fsv32::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o))
        return false;

    auto cp = static_cast<const convolution_params&>(p);

    // Output feature padding must be multiple of fsv to keep block alignment
    if (cp.output.Feature().pad.before % fsv != 0)
        return false;

    return true;
}

JitConstants ConvolutionKernel_bfyx_to_fs_byx_fsv32::GetJitConstants(const convolution_params& params,
                                                                     const DispatchData& kd) const {
    auto jit = ConvolutionKernelBase::GetJitConstants(params, kd);

    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_WIDTH", kd.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_HEIGHT", kd.cldnnStyle.blockHeight));

    auto inputBlockWidth =
        getInputSize(params.stride.x, params.filterSize.x, params.dilation.x, kd.cldnnStyle.blockWidth);
    auto inputBlockHeight =
        getInputSize(params.stride.y, params.filterSize.y, params.dilation.y, kd.cldnnStyle.blockHeight);

    auto inputBlockWidthRound = RoundUp(inputBlockWidth, subGroupSize);

    jit.AddConstant(MakeJitConstant("INPUT_BLOCK_WIDTH", inputBlockWidthRound));
    jit.AddConstant(MakeJitConstant("INPUT_BLOCK_HEIGHT", inputBlockHeight));
    jit.AddConstant(MakeJitConstant("FSV", fsv));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("FSV_PER_THREAD", fsvPerThread));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_vec_elem = {"_VEC_ELEM",
                                               {"b", "fs * FSV + sglid + out_f * SUB_GROUP_SIZE", "or + out_y", "oc + out_x"},
                                               "tmp_write[out_f]", input_dt, 1 };
        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                             {"b", "fs * FSV + sglid + out_f * SUB_GROUP_SIZE", "or + out_y", "oc + out_x"},
                                             "out[out_idx]", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec_elem, conf_scalar}));
    }

    return jit;
}

KernelsData ConvolutionKernel_bfyx_to_fs_byx_fsv32::GetTunedKernelsDataByIndex(const Params& params,
                                                                               const optional_params& options,
                                                                               const int autoTuneIndex) const {
    auto tuneOptions = GetAutoTuneOptions(params, autoTuneIndex);
    return GetCommonKernelsData(params, options, tuneOptions.exeMode, autoTuneIndex);
}

KernelsData ConvolutionKernel_bfyx_to_fs_byx_fsv32::GetKernelsData(const Params& params,
                                                                   const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

KernelsData ConvolutionKernel_bfyx_to_fs_byx_fsv32::GetKernelsDataForAutoTune(const Params& params,
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
