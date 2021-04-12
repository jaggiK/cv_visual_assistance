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

#include "fused_conv_bn_scale_kernel_base.h"
#include <vector>

namespace kernel_selector {

class fused_conv_bn_scale_kernel_ref : public fused_conv_bn_scale_kernel_base {
public:
    using Parent = fused_conv_bn_scale_kernel_base;

    fused_conv_bn_scale_kernel_ref() : fused_conv_bn_scale_kernel_base("fused_conv_bn_scale_kernel_ref") {}
    virtual ~fused_conv_bn_scale_kernel_ref() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    std::vector<WeightsLayout> GetSupportedWeightLayouts(const fused_conv_bn_scale_params&) const override {
        return {
            WeightsLayout::oiyx,
        };
    }
    DispatchData SetDefault(const fused_conv_bn_scale_params& arg) const override;
    JitConstants GetJitConstants(const fused_conv_bn_scale_params& params, const DispatchData& kd) const override;
};
}  // namespace kernel_selector