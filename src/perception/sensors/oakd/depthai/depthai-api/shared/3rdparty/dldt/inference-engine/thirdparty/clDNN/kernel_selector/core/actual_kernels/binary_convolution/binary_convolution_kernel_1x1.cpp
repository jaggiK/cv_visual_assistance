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


#include <iostream>
#include "binary_convolution_kernel_1x1.h"
#include <string>
#include <core/actual_kernels/activation/activation_kernel_base.h>

namespace kernel_selector {

static const int sub_group_size = 16;
static const int ic_pack_size = 32;
static const int xy_block_size = 16;

ParamsKey BinaryConvolutionKernel1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernel1x1::SetDefault(
    const binary_convolution_params& params,
    int) const {
    DispatchData kd = BinaryConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = Align(x * y, sub_group_size);
    kd.gws1 = CeilDiv(f, 2 * sub_group_size);  // 1 WI calcs 32 OC
    kd.gws2 = b;

    kd.lws0 = sub_group_size;
    kd.lws1 = 1;
    kd.lws2 = 1;

    kd.effiency = FORCE_PRIORITY_1;

    return kd;
}

bool BinaryConvolutionKernel1x1::Validate(const Params& p, const optional_params& o) const {
    if (!BinaryConvolutionKernelBase::Validate(p, o) || !CovolutionBinaryCheckInput(p, o))
        return false;

    const auto& params = static_cast<const binary_convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    const bool bOutputSizes = output.X().v != input.X().v || output.Y().v != input.Y().v;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;
    const bool bGroups = params.split > 1 || params.groups > 1 || params.depthwise_separable_opt;

    if (bOutputSizes || bFilterSize || bStride || bGroups)
        return false;

    return true;
}

JitConstants BinaryConvolutionKernel1x1::GetJitConstants(const binary_convolution_params& params,
                                                         const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_NUM_PACKED", CeilDiv(params.inputs[0].Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_NUM_PACKED", CeilDiv(params.output.Feature().v, ic_pack_size)));
    jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", params.output.X().pad.Total() != 0));
    jit.AddConstant(MakeJitConstant("XY_BLOCK_SIZE", xy_block_size));
    if (params.inputs[0].Feature().v % ic_pack_size) {
        jit.AddConstant(MakeJitConstant("LEFTOVERS_IC", params.inputs[0].Feature().v % ic_pack_size));
        jit.AddConstant(MakeJitConstant("FILTER_MASK",
                                        (0xFFFFFFFF >> (ic_pack_size - params.inputs[0].Feature().v % ic_pack_size))));
    }

    if (params.output.GetDType() == Datatype::BINARY) {
        jit.AddConstant(MakeJitConstant("BINARY_PACKED_OUTPUT", 1));
    }

    return jit;
}

JitConstants BinaryConvolutionKernel1x1::GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                                        const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    auto input_dt = GetUnitType(params);
    FusedOpsConfiguration conf = {"", {"b", "(f_block*16 + oc)", "y", "x"}, "res", input_dt, 1 };
    jit.Merge(MakeFusedOpsDeclsJitConstants(params, {conf}));

    size_t op_id = 0;
    std::string input_decls = "";
    std::string eltwise_fused_ops = "";
    std::string prepare_data = "";
    for (auto& fused_dep : params.fused_ops) {
        auto get_aligned_load2 = [&](std::string ptr, std::string byte_offset) -> std::string {
            if (fused_dep.tensors[0].GetDType() == Datatype::F32)
                return "(intel_sub_group_block_read2((const __global uint*)(" + ptr + ") + (" + byte_offset + ")))";
            else
                return "(intel_sub_group_block_read_us2((const __global ushort*)(" + ptr + ") + (" + byte_offset +
                       ")))";
        };

        auto get_shuffle = [&](std::string var, std::string lid) -> std::string {
            return "(intel_sub_group_shuffle(" + var + ", " + lid + "))";
        };

        std::string data_type = fused_dep.GetInputTypeName(0, 1);
        std::string vec_data_type = fused_dep.GetInputTypeName(0, 2);
        std::string sc = "sc" + std::to_string(op_id);
        std::string sh = "sh" + std::to_string(op_id);
        switch (fused_dep.GetType()) {
            case KernelType::SCALE: {
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float2" : "as_half2";
                if (fused_dep.tensors.size() == 1) {
                    std::string var_name = fused_dep.GetInputVarName(0);
                    prepare_data += "\\\n\t" + vec_data_type + " " + var_name + " = " + cast_type +
                                    get_aligned_load2(fused_dep.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + sc + " = (oc < 16) ? " +
                                    get_shuffle(var_name + ".s0", "oc") + " : " + get_shuffle(var_name + ".s1", "oc") + ";";
                    eltwise_fused_ops += "\\\n\tres = res*" + sc + ";";
                } else {
                    std::string var0_name = fused_dep.GetInputVarName(0);
                    std::string var1_name = fused_dep.GetInputVarName(1);
                    prepare_data += "\\\n\t" + vec_data_type + " " + var0_name + " = " + cast_type +
                                    get_aligned_load2(fused_dep.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    prepare_data += "\\\n\t" + vec_data_type + " " + var1_name + " = " + cast_type +
                                    get_aligned_load2(fused_dep.GetInputPtrName(1), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + sc +" = (oc < 16) ? " +
                                    get_shuffle(var0_name + ".s0", "oc") + " : " + get_shuffle(var0_name + ".s1", "oc") + ";";
                    eltwise_fused_ops += "\\\n\t" + data_type + " " + sh + " = (oc < 16) ? " +
                                    get_shuffle(var1_name + ".s0", "oc") + " : " + get_shuffle(var1_name + ".s1", "oc") + ";";
                    eltwise_fused_ops += "\\\n\tres = res*" + sc + " + " + sh + ";";
                }

                break;
            }
            case KernelType::QUANTIZE: {
                std::string var_name_in = fused_dep.GetInputVarName(0);
                std::string var_name_out = fused_dep.GetInputVarName(3);
                std::string cast_type_vec = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float2" : "as_half2";
                std::string cast_type = (fused_dep.tensors[0].GetDType() == Datatype::F32) ? "as_float" : "as_half";

                prepare_data += "\\\n\tint packed_res = 0;";
                if (fused_dep.tensors[0].Feature().v == params.output.Feature().v) {
                    prepare_data += "\\\n\t" + vec_data_type + " " + var_name_in + " = " + cast_type_vec +
                                    get_aligned_load2(fused_dep.GetInputPtrName(0), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops += "\\\n\t" + data_type + " thresh = (oc < 16) ? " + get_shuffle(var_name_in + ".s0", "oc") +
                                         " : " + get_shuffle(var_name_in + ".s1", "oc") + ";";
                } else {
                    prepare_data += "\\\n\t" + data_type + " " + var_name_in + " = " + cast_type +
                                 + "(" + fused_dep.GetInputPtrName(0) + "[0]);";
                    eltwise_fused_ops += "\\\n\t" + data_type + " thresh = " + var_name_in + ";";
                }


                if (fused_dep.tensors[2].Feature().v == params.output.Feature().v) {
                    // Per-channel output value
                    prepare_data += "\\\n\t" + vec_data_type + " " + var_name_out + " = " + cast_type_vec +
                                    get_aligned_load2(fused_dep.GetInputPtrName(3), "f_block*OC_BLOCK_SIZE") + ";";
                    eltwise_fused_ops +="\\\n\t" + data_type + " out_val = (oc < 16) ? " + get_shuffle(var_name_out + ".s0", "oc") +
                                         " : " + get_shuffle(var_name_out + ".s1", "oc") + ";";
                } else {
                    // Per-tensor output value
                    prepare_data += "\\\n\t" + data_type + " " + var_name_out + " = " + cast_type +
                                    + "(" + fused_dep.GetInputPtrName(3) + "[0]);";
                    eltwise_fused_ops += "\\\n\t" + data_type + " out_val = " + var_name_out + ";";
                }
                eltwise_fused_ops += "\\\n\tif (out_val == 1) ";
                eltwise_fused_ops += "\\\n\t\tpacked_res |= (res > thresh) << oc;";
                eltwise_fused_ops += "\\\n\telse ";
                eltwise_fused_ops += "\\\n\t\tpacked_res |= (res <= thresh) << oc;";

                break;
            }
            case KernelType::ACTIVATION: {
                auto p = fused_dep.GetOpParams<activation_fuse_params>();
                base_activation_params activation = p->param;
                if (activation.function != ActivationFunction::NONE) {
                    auto suffix = "_FUSED_OP" + std::to_string(op_id);

                    jit.Merge(MakeActivationJitConstants(activation, fused_dep.output_tensor.GetDType(), suffix));
                    eltwise_fused_ops += "\\\n\tres = ACTIVATION" + suffix + "((OUTPUT_TYPE)res, ACTIVATION_PARAMS" + suffix + ");";
                }
                break;
            }
            default:
                throw std::invalid_argument("Invalid fused op in binary_convolution kernel: " + params.layerID);
        }

        op_id++;
    }
    jit.AddConstant(MakeJitConstant("DO_ELTWISE_FUSED_OPS", eltwise_fused_ops));
    jit.AddConstant(MakeJitConstant("FUSED_OPS_PREPARE_DATA", prepare_data));
    jit.AddConstant(MakeJitConstant("CUSTOM_FUSED_OPS", true));

    return jit;
}

KernelsData BinaryConvolutionKernel1x1::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
