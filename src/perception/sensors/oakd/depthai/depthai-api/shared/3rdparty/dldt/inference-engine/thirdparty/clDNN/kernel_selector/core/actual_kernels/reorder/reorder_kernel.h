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

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernelRef : public ReorderKernelBase {
public:
    ReorderKernelRef() : ReorderKernelBase("reorder_data") {}
    virtual ~ReorderKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const reorder_params& params) const override;
};
}  // namespace kernel_selector
