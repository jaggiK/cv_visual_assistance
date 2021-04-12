/*
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
*/

#pragma once

#include "common_kernel_base.h"
#include "weight_bias_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightsBiasKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class WeightBiasKernelBase : public common_kernel_base {
public:
    using common_kernel_base::common_kernel_base;
    virtual ~WeightBiasKernelBase() {}

protected:
    virtual JitConstants GetJitConstants(const weight_bias_params& params) const;
};
}  // namespace kernel_selector