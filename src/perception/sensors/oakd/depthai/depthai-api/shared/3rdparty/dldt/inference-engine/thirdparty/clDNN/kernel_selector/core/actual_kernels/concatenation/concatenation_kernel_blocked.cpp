﻿// Copyright (c) 2016 Intel Corporation
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


#include <iostream>
#include "concatenation_kernel_blocked.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ConcatenationKernelBlocked::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx_f16);
    k.EnableOutputLayout(DataLayout::bfyx_f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatKernelPerInput();
    return k;
}

bool ConcatenationKernelBlocked::Validate(const Params& p, const optional_params& o) const {
    if (!ConcatenationKernelBase::Validate(p, o)) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    // all inputs have to have same layout
    auto same_layout = params.inputs[0].GetLayout();
    for (const auto& lt : params.inputs) {
        if (lt.GetLayout() != same_layout) {
            return false;
        }
    }

    if (params.axis != ConcatAxis::FEATURE)
        return false;

    return true;
}

ConcatenationKernelBase::DispatchData ConcatenationKernelBlocked::SetDefault(const concatenation_params& params) const {
    DispatchData runInfo = ConcatenationKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];

    runInfo.gws0 = input.Batch().v;
    runInfo.gws1 = Align(input.Feature().v, 16);
    runInfo.gws2 = input.X().v * input.Y().v;

    runInfo.lws0 = 1;
    runInfo.lws1 = 16;
    runInfo.lws2 = 1;

    runInfo.effiency = FORCE_PRIORITY_1;

    return runInfo;
}

JitConstants ConcatenationKernelBlocked::GetJitConstants(const concatenation_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("ALIGNED", params.isAligned));

    return jit;
}

KernelsData ConcatenationKernelBlocked::GetKernelsData(const Params& params, const optional_params& optParams) const {
    return GetCommonKernelsData(params, optParams);
}
}  // namespace kernel_selector
