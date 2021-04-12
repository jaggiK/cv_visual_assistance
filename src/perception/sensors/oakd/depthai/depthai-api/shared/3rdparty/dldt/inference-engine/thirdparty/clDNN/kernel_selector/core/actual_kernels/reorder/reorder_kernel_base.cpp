﻿// Copyright (c) 2016-2019 Intel Corporation
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


#include "kernel_selector_common.h"
#include "reorder_kernel_base.h"
#include "common_tools.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
inline uint32_t SubGroupSize(WeightsLayout l) {
    switch (l) {
        case WeightsLayout::os_iyx_osv16:
        case WeightsLayout::os_iyx_osv32:
        case WeightsLayout::os_iyx_osv64:
        case WeightsLayout::os_iyx_osv16_rotate_180:
        case WeightsLayout::os_i_osv16:
        case WeightsLayout::os_i_osv16__ai8:
        case WeightsLayout::i_yxs_os_yxsv2_osv16:
        case WeightsLayout::iy_xs_os_xsv2_osv16__ao32:
        case WeightsLayout::os_is_yx_osv32_isv32p:
        case WeightsLayout::o_i_yx_i16_o16:
        case WeightsLayout::oiyx_o16:
        case WeightsLayout::o_i_zyx_i16_o16:
        case WeightsLayout::i_o_zyx_o16_i16:
        case WeightsLayout::o_i_zyx_i8_o16_i2:
        case WeightsLayout::ozyxi_o16:
            return 16;
        case WeightsLayout::os_i_osv8__ai8:
        case WeightsLayout::iy_xs_os_xsv2_osv8__ao32:
            return 8;
        default:
            return 1;
    }
}

inline uint32_t SubGroupSize(DataLayout l) {
    switch (l) {
        case DataLayout::bs_f_bsv16__af8:
            return 16;
        case DataLayout::bs_f_bsv8__af8:
            return 8;
        default:
            return 1;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeReorderWeightsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline JitConstants MakeReorderWeightsJitConstants(const reorder_weights_params& params) {
    const auto& input = params.input;
    const auto& output = params.output;
    const bool fp16Supported = output.GetDType() == WeightsType::F16 || input.GetDType() == WeightsType::F16;

    JitConstants jit{
        MakeJitConstant("FP16_SUPPORTED", fp16Supported),  // TODO: use engine
        MakeJitConstant("FP16_UNIT_USED", fp16Supported),
        MakeJitConstant("INPUT0", input),
        MakeJitConstant("OUTPUT", output),
    };

    if (fp16Supported) {
        jit.Merge(MakeUnitTypeJitConstants(Datatype::F16));
    } else {
        jit.Merge(MakeUnitTypeJitConstants(Datatype::F32));
    }
    return jit;
}

JitConstants ReorderKernelBase::GetJitConstants(const reorder_weights_params& params) const {
    JitConstants mem_consts = MakeReorderWeightsJitConstants(params);

    mem_consts.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", SubGroupSize(params.output.GetLayout())));

    return mem_consts;
}

JitConstants ReorderKernelBase::GetJitConstants(const reorder_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("MEAN_SUBTRACT_" + toString(params.mode), 1));

    if (params.mode == MeanSubtractMode::INSIDE_PARAMS) {
        jit.AddConstant(MakeJitConstant("VALUE_TO_SUBTRACT", params.meanValues));
        jit.AddConstant(MakeJitConstant("TO_MEAN_TYPE", "convert_float"));
    } else if (params.mode == MeanSubtractMode::IN_BUFFER) {
        jit.AddConstant(MakeJitConstant("MEAN_SUBTRACT", params.mean));
        jit.AddConstant(MakeJitConstant("TO_MEAN_TYPE", "convert_" + toCLType(params.mean.GetDType())));
    }

    // Type JITs:

    // half->half without subtraction and activation (so plain reorder) can be done on shorts without explicit fp16 support
    bool useUshort = (params.inputs[0].GetDType() == Datatype::F16 && params.output.GetDType() == Datatype::F16 &&
                      params.mode == MeanSubtractMode::NONE && params.activations.empty());

    Datatype calc_type = useUshort ? Datatype::UINT16 : params.inputs[0].GetDType();
    Datatype output_reorder_type = useUshort ? Datatype::UINT16 : params.output.GetDType();
    Datatype input_reorder_type = useUshort ? Datatype::UINT16 : params.inputs[0].GetDType();

    jit.Merge(MakeTypeJitConstants(calc_type, "CALC"));
    jit.Merge(MakeTypeJitConstants(input_reorder_type, "INPUT_REORDER"));
    jit.Merge(MakeTypeJitConstants(output_reorder_type, "OUTPUT_REORDER"));

    jit.AddConstant(MakeJitConstant("MEAN_OP(val, mean_val)", getMeanOpString(params.mean_op)));

    // Type parametrized activation:
    jit.Merge(MakeActivationJitConstants(params.activations, GetUnitType(params), "_TYPED", true));

    // TODO: Move to lower classes
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", SubGroupSize(params.output.GetLayout())));

    return jit;
}

ReorderKernelBase::DispatchData ReorderKernelBase::SetDefault(const reorder_weights_params& params) const {
    const auto& out = params.output;

    DispatchData kd;

    std::vector<size_t> global(3);

    global = {out.OFM().v, out.IFM().v, out.X().v * out.Y().v * out.Z().v};
    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

ReorderKernelBase::DispatchData ReorderKernelBase::SetDefault(const reorder_params& params) const {
    DispatchData kd;

    auto global = GetTensorFriendlyWorkGroups(params.inputs[0]);
    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        std::vector<size_t> sizes = { 32, 16, 8, 4 };
        for (auto& s : sizes) {
            if (kd.gws2 % s == 0) {
                kd.lws0 = 1;
                kd.lws1 = 1;
                kd.lws2 = s;
                break;
            }
        }
    }

    return kd;
}

KernelsData ReorderKernelBase::GetCommonKernelsData(const reorder_weights_params& params, const optional_params& options, float estimated_time) const {
    assert(params.GetType() == KernelType::REORDER);

    KernelData kd = KernelData::Default<reorder_weights_params>(params);
    reorder_weights_params& newParams = *static_cast<reorder_weights_params*>(kd.params.get());

    DispatchData runInfo;

    runInfo = SetDefault(newParams);

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kernel.arguments = GetArgsDesc(1, false, false);

    kd.estimatedTime = estimated_time;

    return {kd};
}

KernelsData ReorderKernelBase::GetCommonKernelsData(const reorder_params& params, const optional_params& options, float estimated_time) const {
    if (!Validate(params, options)) {
        return {};
    }
    assert(params.GetType() == KernelType::REORDER);

    KernelData kd = KernelData::Default<reorder_params>(params);
    reorder_params& newParams = *static_cast<reorder_params*>(kd.params.get());

    DispatchData runInfo;

    runInfo = SetDefault(newParams);

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kernel.arguments = GetArgsDesc(1, false, false);
    if (newParams.mode == MeanSubtractMode::IN_BUFFER) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::BIAS, 0});
    }

    kd.estimatedTime = estimated_time;

    return {kd};
}
}  // namespace kernel_selector
