﻿// Copyright (c) 2017 Intel Corporation
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

#include "convolution_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {
class ConvolutionKernel_bfyx_3x3_dw_opt : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;
    ConvolutionKernel_bfyx_3x3_dw_opt();
    virtual ~ConvolutionKernel_bfyx_3x3_dw_opt() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                                   const optional_params& options,
                                                   int autoTuneIndex) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params&, const optional_params&) const override;
    std::vector<WeightsLayout> GetSupportedWeightLayouts(const convolution_params&) const override {
        return {WeightsLayout::oiyx};
    }
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& kd) const override;
    DispatchData SetDefault(const convolution_params& params, int autoTuneIndex = -1) const override;

    struct AutoTuneOption {
        stSize tileDims;
        std::string exeMode;
    };

    AutoTuneOption GetAutoTuneOptions(const Params& arg, int autoTuneIndex) const;
    std::vector<AutoTuneOption> autoTuneOptions = {};
};
}  // namespace kernel_selector