/*
// Copyright (c) 2018 Intel Corporation
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

#include "region_yolo_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

ParamsKey RegionYoloKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants RegionYoloKernelRef::GetJitConstants(const region_yolo_params& ry) const {
    JitConstants jit = MakeBaseParamsJitConstants(ry);

    jit.AddConstants({MakeJitConstant("COORDS", ry.coords),
                      MakeJitConstant("CLASSES", ry.classes),
                      MakeJitConstant("NUM", ry.num),
                      MakeJitConstant("DO_SOFTMAX", ry.do_softmax),
                      MakeJitConstant("MASK_SIZE", ry.mask_size)});

    return jit;
}

RegionYoloKernelRef::DispatchData SetDefault(const region_yolo_params& params) {
    RegionYoloKernelRef::DispatchData kd;

    kd.fp16UnitUsed = (params.inputs[0].GetDType() == Datatype::F16);

    const auto& input = params.inputs[0];
    std::vector<size_t> global;
    if (input.GetLayout() == DataLayout::bfyx) {
        global = {input.X().v * input.Y().v, 1, 1};
    } else {
        global = {input.Feature().v * input.Batch().v, input.X().v, input.Y().v};
    }
    // Determine global work sizes.
    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}
KernelsData RegionYoloKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::REGION_YOLO);
    const region_yolo_params& orgParams = static_cast<const region_yolo_params&>(params);

    DispatchData runInfo = SetDefault(orgParams);
    KernelData kd = KernelData::Default<region_yolo_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = FORCE_PRIORITY_9;

    return {kd};
}
}  // namespace kernel_selector
