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


#pragma once

#include "fused_conv_eltwise_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class fused_conv_eltwise_kernel_bfyx_1x1_opt : public fused_conv_eltwise_kernel_base {
public:
    using Parent = fused_conv_eltwise_kernel_base;
    fused_conv_eltwise_kernel_bfyx_1x1_opt() : fused_conv_eltwise_kernel_base("fused_conv_eltwise_gpu_bfyx_1x1_opt") {}

    virtual ~fused_conv_eltwise_kernel_bfyx_1x1_opt() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<WeightsLayout> GetSupportedWeightLayouts(const fused_conv_eltwise_params&) const override;
    std::string GetKernelName(const fused_conv_eltwise_params& params) const override;
    bool NeedPaddedInput() const override { return true; }
    JitConstants GetJitConstants(const fused_conv_eltwise_params& params, const DispatchData& kd) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const fused_conv_eltwise_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector