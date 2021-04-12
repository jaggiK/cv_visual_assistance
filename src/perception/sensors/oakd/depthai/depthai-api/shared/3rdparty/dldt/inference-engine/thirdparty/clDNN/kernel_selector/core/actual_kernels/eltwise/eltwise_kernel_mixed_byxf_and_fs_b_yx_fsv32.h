﻿// Copyright (c) 2019 Intel Corporation
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

#include "eltwise_kernel_base.h"

/*
    This kernel is basicaly a eltwise_kernel_vload8 but GetKernelsData is modfied
    to roundup features number to 32 when
*/

namespace kernel_selector {
class EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32 : public EltwiseKernelBase {
public:
    EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32() : EltwiseKernelBase("eltwise_mixed_byxf_and_fs_b_yx_fsv32") {}
    virtual ~EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
};
}  // namespace kernel_selector