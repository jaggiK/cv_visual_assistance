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


#include "mvn_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants MVNKernelBase::GetJitConstants(const mvn_params& params, MVNKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("EPSILON", params.epsilon),
        MakeJitConstant(toString(params.mvnMode), ""),
        MakeJitConstant("NORMALIZE_VARIANCE", params.mvnNormalizeVariance),
    });

    return jit;
}

MVNKernelBase::DispatchData MVNKernelBase::SetDefault(const mvn_params& params) const {
    const auto& output = params.output;

    DispatchData kd;

    std::vector<size_t> global(3);

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        global = {output.Batch().v, output.Feature().v, 1};
    } else {
        global = {output.Batch().v, 1, 1};
    }

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData MVNKernelBase::GetCommonKernelsData(const Params& params,
                                                const optional_params& options,
                                                float estimated_time) const {
    assert(params.GetType() == KernelType::MVN);

    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    DispatchData runInfo;

    runInfo = SetDefault(orgParams);

    KernelData kd = KernelData::Default<mvn_params>(params);

    auto finalKernelName = GetKernelName(orgParams);
    auto cldnn_jit = GetJitConstants(orgParams, runInfo);
    auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
    auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, finalKernelName, jit, entry_point);

    kd.estimatedTime = estimated_time;

    return {kd};
}
}  // namespace kernel_selector
