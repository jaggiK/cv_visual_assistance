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


#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernel_to_yxfb_batched : public ReorderKernelBase {
public:
    ReorderKernel_to_yxfb_batched() : ReorderKernelBase("reorder_data_to_yxfb_batched") {}
    ParamsKey GetSupportedKey() const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;

protected:
    JitConstants GetJitConstants(const reorder_params& params) const override;
    DispatchData SetDefault(const reorder_params& arg) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector