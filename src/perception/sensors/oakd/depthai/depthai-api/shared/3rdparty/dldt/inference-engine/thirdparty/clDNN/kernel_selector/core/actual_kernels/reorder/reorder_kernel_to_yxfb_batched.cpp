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


#include "reorder_kernel_to_yxfb_batched.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderKernel_to_yxfb_batched::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool ReorderKernel_to_yxfb_batched::Validate(const Params& params, const optional_params& o) const {
    if (!ReorderKernelBase::Validate(params, o)) {
        return false;
    }

    const reorder_params& r_params = static_cast<const reorder_params&>(params);

    const auto& output = r_params.output;
    // output cannot have padding for this implementation
    if (output.X().pad.Total() != 0 || output.Y().pad.Total() != 0 || output.Feature().pad.Total() != 0 ||
        output.Batch().pad.Total() != 0) {
        return false;
    }

    if ((r_params.inputs[0].GetLayout() == DataLayout::bfzyx_f16 || r_params.inputs[0].GetLayout() == DataLayout::bfzyx_b16f16) &&
        r_params.inputs[0].Z().v != 1)
        return false;

    return true;
}

JitConstants ReorderKernel_to_yxfb_batched::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    KernelData kd = KernelData::Default<reorder_params>(params);
    reorder_params& newParams = *static_cast<reorder_params*>(kd.params.get());

    const auto& input = newParams.inputs[0];
    jit.AddConstant(MakeJitConstant("ELEMENTS_COUNT", input.LogicalSize()));

    return jit;
}

ReorderKernelBase::DispatchData ReorderKernel_to_yxfb_batched::SetDefault(const reorder_params& params) const {
    DispatchData kd;

    const auto& input = params.inputs[0];

    unsigned int gws = (unsigned int)input.LogicalSize();

    kd.gws0 = Align(gws, 8 * input.Batch().v) / input.Batch().v;
    kd.gws1 = 1;
    kd.gws2 = 1;

    kd.lws0 = 8;
    kd.lws1 = 1;
    kd.lws2 = 1;

    return kd;
}

KernelsData ReorderKernel_to_yxfb_batched::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    auto estimatedTime = FORCE_PRIORITY_1;

    // this algorithm is designed to perform well on multiple batches, so if batch is 1, then don't use it
    if (orgParams.inputs[0].Batch().v == 1)
        estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return GetCommonKernelsData(orgParams, options, estimatedTime);
}
}  // namespace kernel_selector