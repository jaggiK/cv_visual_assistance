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

#include "convolution_kernel_mmad_bfyx_to_b_fs_yx_fsv4.h"
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>

namespace kernel_selector {

ParamsKey ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.DisableTuning();
    return k;
}

bool ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::Validate(const Params &p, const optional_params &o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if (params.inputs[0].Feature().v != 3)
        return false;

    return true;
}

ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::AutoTuneOption ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::GetAutoTuneOptions(const Params &p,
                                                                                                                        int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    AutoTuneOption option = {0, 0, 0, DEFAULT};

    auto &params = dynamic_cast<const convolution_params &>(p);
    auto &output = params.output;

    // TODO: Check if other block size can improve performance
    option.blockHeight = 1;
    option.prefetch = 1;
    if (output.LogicalSize() < 49 * 1024) {
        option.blockWidth = 4;
    } else {
        option.blockWidth = 8;
    }

    return option;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::SetDefault(const convolution_params &cp,
                                                                                          int autoTuneIndex) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(cp);

    auto tuneOptions = GetAutoTuneOptions(cp, autoTuneIndex);
    runInfo.cldnnStyle.blockWidth = tuneOptions.blockWidth;
    runInfo.cldnnStyle.blockHeight = tuneOptions.blockHeight;
    runInfo.cldnnStyle.prefetch = tuneOptions.prefetch;

    runInfo.effiency = FORCE_PRIORITY_3;

    runInfo.gws0 = Align(cp.output.Feature().v, 32) / 2;
    runInfo.gws1 = CeilDiv(cp.output.X().v, runInfo.cldnnStyle.blockWidth) * cp.output.Y().v;
    runInfo.gws2 = cp.output.Batch().v;

    runInfo.lws0 = 16;
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    return runInfo;
}

JitConstants ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::GetJitConstants(const convolution_params &params,
                                                                        const DispatchData &runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws0));
    jit.AddConstant(MakeJitConstant("OSV", 32));
    jit.AddConstant(MakeJitConstant("ISV", 32));
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", runInfo.cldnnStyle.blockWidth));
    jit.AddConstant(MakeJitConstant("IFM_BLOCKS", CeilDiv(params.inputs[0].Feature().v, 32)));
    auto input = params.inputs[0];
    auto output = params.output;
    auto blockWidth = runInfo.cldnnStyle.blockWidth;
    size_t input_line_size = std::min(params.stride.x * (blockWidth - 1) + (params.weights.X().v - 1) * params.dilation.x + 1,
                                      input.X().v + input.X().pad.Total());

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("INPUT_LINE_SIZE", input_line_size));

    jit.Merge(MakeTypeJitConstants(GetPackedInputType(params), "PACKED_IN"));
    jit.Merge(MakeTypeJitConstants(GetPackedType(params.output.GetDType(), 2), "PACKED_OUT"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf0 = {"_0", {"b", "(fg*32 + 2*lid+0)", "y", "(x+i)"}, "res0", input_dt, 1};
        FusedOpsConfiguration conf1 = {"_1", {"b", "(fg*32 + 2*lid+1)", "y", "(x+i)"}, "res1", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf0, conf1}));
    }

    return jit;
}

KernelsData ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::GetKernelsData(const Params &params, const optional_params &options) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params, options);
    if (!kd.empty()) {
        kd[0].estimatedTime = FORCE_PRIORITY_2;
    }

    return kd;
}

KernelsData ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4::GetKernelsDataForAutoTune(const Params &params,
                                                                                 const optional_params &options) const {
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
