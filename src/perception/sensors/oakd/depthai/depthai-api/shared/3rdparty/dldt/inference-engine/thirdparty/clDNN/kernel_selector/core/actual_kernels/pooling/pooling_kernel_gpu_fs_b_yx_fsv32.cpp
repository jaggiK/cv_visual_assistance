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


#include "pooling_kernel_gpu_fs_b_yx_fsv32.h"

namespace kernel_selector {
ParamsKey PoolingKerneGPU_fs_b_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
    return k;
}

PoolingKernelBase::DispatchData PoolingKerneGPU_fs_b_yx_fsv32::SetDefault(const pooling_params& params) const {
    DispatchData runInfo = PoolingKernelBase::SetDefault(params);

    runInfo.gws0 = params.output.X().v;  // X output blocks
    runInfo.gws1 = params.output.Y().v;  // Y output clocks
    // in fs_b_yx_fsv32 format we will process 2 features per work item, so reads/writes are done in full writes for
    // fp16
    runInfo.gws2 = RoundUp(params.output.Feature().v, 32) * params.output.Batch().v / 2;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = 16;

    return runInfo;
}

bool PoolingKerneGPU_fs_b_yx_fsv32::Validate(const Params& p, const optional_params& o) const {
    if (!PoolingKernelBase::Validate(p, o))
        return false;

    auto pp = static_cast<const pooling_params&>(p);

    // Feature padding before must be aligned to 32 to keep slices aligned
    if (pp.output.Feature().pad.before % 32 != 0)
        return false;

    return true;
}

JitConstants PoolingKerneGPU_fs_b_yx_fsv32::GetJitConstants(const pooling_params& params, DispatchData kd) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, kd);
    auto pp = static_cast<const pooling_params&>(params);

    // Heurestic needed for very big pool size.
    // ToDo Can it be changed to lower pool sizes?
    if (pp.poolSize.x >= 7 && pp.poolSize.y >= 7 && pp.poolType == PoolType::AVG) {
        jit.AddConstant(MakeJitConstant("USE_FLOAT_ACC", true));
    }

    return jit;
}

KernelsData PoolingKerneGPU_fs_b_yx_fsv32::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}
}  // namespace kernel_selector
