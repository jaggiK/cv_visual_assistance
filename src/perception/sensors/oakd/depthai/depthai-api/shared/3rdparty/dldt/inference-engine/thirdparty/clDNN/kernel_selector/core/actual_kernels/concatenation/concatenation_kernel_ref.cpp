﻿// Copyright (c) 2016 Intel Corporation
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


#include "concatenation_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>

namespace kernel_selector {

ParamsKey ConcatenationKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::fb);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::fyxb);
    k.EnableInputLayout(DataLayout::bfyx_f16);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fyxb);
    k.EnableOutputLayout(DataLayout::bfyx_f16);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::X);
    k.EnableConcatAxis(ConcatAxis::Y);
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatAxis(ConcatAxis::BATCH);
    k.EnableConcatKernelPerInput();
    k.EnableDifferentTypes();
    return k;
}

JitConstants ConcatenationKernelRef::GetJitConstants(const concatenation_params& params) const {
    auto cldnnJit = ConcatenationKernelBase::GetJitConstants(params);
    auto input_format = params.inputs[0].GetLayout();

    if (params.inputs[0].Feature().v != 1) {
        cldnnJit.AddConstant(MakeJitConstant("CHECK_FEATURES", 1));
        int f_channel = DataTensor::Channelndex(params.output.GetLayout(), Tensor::DataChannelName::FEATURE);
        cldnnJit.AddConstant(MakeJitConstant("FEATURE_CHANNEL", f_channel));
    }

    // default values when input_format = output_format
    // d3 = batch, d2 = feature, d1 = y, d0 = x
    std::vector<std::string> dims_id = {"d3", "d2", "d1", "d0"};
    auto axis = ConcatenationKernelBase::GetConcatChannel(params);

    std::vector<Tensor::DataChannelName> axis_order = { Tensor::DataChannelName::BATCH,
                                                        Tensor::DataChannelName::FEATURE,
                                                        Tensor::DataChannelName::Y,
                                                        Tensor::DataChannelName::X };

    std::string input_dims_order = "";
    std::string output_dims_order = "";
    for (size_t i = 0; i < dims_id.size(); i++) {
        input_dims_order += dims_id[i] + (i == dims_id.size() - 1 ? "" : ",");
        if (axis_order[i] == axis)
            output_dims_order += "(" + dims_id[i] + " + output_offset_in_concat_axis)" +
                                 (i == dims_id.size() - 1 ? "" : ",");
        else
            output_dims_order += dims_id[i] + (i == dims_id.size() - 1 ? "" : ",");
    }

    cldnnJit.AddConstant(MakeJitConstant("INPUT_DIMS_ORDER", input_dims_order));
    cldnnJit.AddConstant(MakeJitConstant("OUTPUT_DIMS_ORDER", output_dims_order));

    cldnnJit.AddConstant(MakeJitConstant("INPUT_DIM_0", DataTensor::Channelndex(input_format, Tensor::DataChannelName::X)));

    return cldnnJit;
}

KernelsData ConcatenationKernelRef::GetKernelsData(const Params& params, const optional_params& optParams) const {
    KernelsData kd = GetCommonKernelsData(params, optParams);

    if (!kd.empty()) {
        for (int i = 0; i < static_cast<int>(kd[0].kernels.size()); i++) {
            auto& kernel = kd[0].kernels[i];

            // to avoid cases when we execute with local work sizes 1x1x1
            if (kernel.workGroups.local[0] == 1 && kernel.workGroups.global[1] != 1) {
                kernel.workGroups.global[1] = Align(kernel.workGroups.global[1], 32);
                kernel.workGroups.local[1] = 32;
            }
        }
    }

    return kd;
}
}  // namespace kernel_selector
