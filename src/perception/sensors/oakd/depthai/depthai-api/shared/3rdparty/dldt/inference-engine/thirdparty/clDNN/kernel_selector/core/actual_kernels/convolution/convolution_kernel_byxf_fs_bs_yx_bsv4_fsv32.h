// Copyright (c) 2016 Intel Corporation
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
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_byxf_fs_bs_yx_bsv4_fsv32 : public ConvolutionKernelBase {
public:
    ConvolutionKernel_byxf_fs_bs_yx_bsv4_fsv32() : ConvolutionKernelBase("convolution_gpu_byxf_fs_bs_yx_bsv4_fsv32") {}
    virtual ~ConvolutionKernel_byxf_fs_bs_yx_bsv4_fsv32() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    ConvolutionKernelBase::DispatchData SetDefault(const convolution_params& arg, int) const override;
    std::vector<WeightsLayout> GetSupportedWeightLayouts(const convolution_params&) const override {
        return {WeightsLayout::yxio};
    }
};
}  // namespace kernel_selector