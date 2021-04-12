/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include <string>
#include <memory>
#include <cstddef>
#include <limits>
#include "common_types.h"
#include "tensor_type.h"
#include "document.h"
#include <vector>

namespace kernel_selector {
using DataTensor = Tensor::DataTensor;
using WeightsTensor = Tensor::WeightsTensor;
using DataLayout = Tensor::DataLayout;
using WeightsLayout = Tensor::WeightsLayout;
using MultiDataTensor = std::vector<DataTensor>;

class JitConstants;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fuse_params {
    virtual ~fuse_params() {}

    KernelType GetType() const { return kType; }
protected:
    explicit fuse_params(KernelType kt) : kType(kt) {}
    KernelType kType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ParamsKey
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ParamsKey {
public:
    ParamsKey() {
        key.restrict.raw = 0;
        key.enableTuning = 1;
        key.machineInfo.raw = 0;
        key.inputType.raw = 0;
        key.outputType.raw = 0;
        key.inputWeightsType.raw = 0;
        key.outputWeightsType.raw = 0;
        key.inputLayout = 0;
        key.outputLayout = 0;
        key.weightsInputLayout = 0;
        key.weightsOutputLayout = 0;
    }

    struct Key {
        union restrict_t {
            struct val_t {
                uint32_t different_types : 1;
                uint32_t different_input_weights_types : 1;
                uint32_t offset : 1;
                uint32_t pitches : 1;
                uint32_t batching : 1;
                uint32_t biasPerFeatureMap : 1;
                uint32_t biasPerOutput : 1;
                uint32_t nonBias : 1;
                uint32_t activationAdditionalParamsAsInput : 1;
                uint32_t FP16Emulation : 1;
                uint32_t gradient : 1;
                uint32_t momentum : 1;
                uint32_t quantization : 1;
                uint32_t output_calibration : 1;
                uint32_t sym_quantization : 1;
                uint32_t asym_w_quantization : 1;
                uint32_t asym_d_quantization : 1;

                union dedicated_t {
                    struct lookt_t {
                        uint32_t axisX : 1;
                        uint32_t axisY : 1;
                        uint32_t axisFeature : 1;
                        uint32_t axisBatch : 1;
                        uint32_t axisXYF : 1;
                        uint32_t indicesF32 : 1;
                        uint32_t indicesOther : 1;
                    } lookt;
                    struct argm_t {
                        uint32_t axisX : 1;
                        uint32_t axisY : 1;
                        uint32_t axisZ : 1;
                        uint32_t axisFeature : 1;
                        uint32_t axisBatch : 1;
                        uint32_t axisXYF : 1;
                    } argm;
                    struct idxsel_t {
                        uint32_t axisX : 1;
                        uint32_t axisY : 1;
                        uint32_t axisFeature : 1;
                        uint32_t axisBatch : 1;
                    } idxsel;
                    struct norm_t {
                        uint32_t across : 1;
                        uint32_t within : 1;
                        uint32_t fixedKenrelDivider : 1;
                        uint32_t dynamicKenrelDivider : 1;
                    } norm;
                    struct mvn_t {
                        uint32_t across : 1;
                        uint32_t within : 1;
                        uint32_t normalize_variance : 1;
                    } mvn;
                    struct pooling_t {
                        uint32_t max : 1;
                        uint32_t avg : 1;
                        uint32_t floor : 1;
                        uint32_t max_with_argmax : 1;
                        uint32_t ceil : 1;
                        uint32_t bilinear : 1;
                        uint32_t deformable_bilinear : 1;
                        uint32_t fixedKenrelDivider : 1;
                        uint32_t dynamicKenrelDivider : 1;
                        uint32_t dynamicKenrelDividerWithPadding : 1;
                        uint32_t position_sensitive : 1;
                    } pooling;
                    struct conv_t {
                        uint32_t split : 1;
                        uint32_t dilation : 1;
                        uint32_t depthwise_separable_opt : 1;
                        uint32_t transposed : 1;
                        uint32_t local : 1;
                        uint32_t grouped : 1;
                        uint32_t deformable : 1;
                    } conv;
                    struct fc_t {
                    } fc;
                    struct softmax_t {
                        uint32_t dimX : 1;
                        uint32_t dimY : 1;
                        uint32_t dimFeature : 1;
                    } softmax;
                    struct region_yolo_t {
                        uint32_t dimX : 1;
                        uint32_t dimY : 1;
                        uint32_t dimFeature : 1;
                        uint32_t coords : 1;
                        uint32_t classes : 1;
                        uint32_t num : 1;
                    } region_yolo;
                    struct reorg_yolo_t {
                        uint32_t dimX : 1;
                        uint32_t dimY : 1;
                        uint32_t dimFeature : 1;
                        uint32_t stride : 1;
                    } reorg_yolo;
                    struct concat_t {
                        uint32_t axisX : 1;
                        uint32_t axisY : 1;
                        uint32_t axisZ : 1;
                        uint32_t axisW : 1;
                        uint32_t axisFeature : 1;
                        uint32_t axisBatch : 1;
                        uint32_t kernelPerInput : 1;
                        uint32_t oneKernel : 1;
                    } concat;
                    struct upsample_t {
                        uint32_t nearest_neighbor : 1;
                        uint32_t caffe_bilinear_interp : 1;
                        uint32_t bilinear_interp : 1;
                    } resample;
                    struct reorder_t {
                        uint32_t winograd : 1;
                    } reorder;
                    struct eltwise_t {
                        uint32_t stride : 1;
                        uint32_t broadcast : 1;
                        uint32_t inputs_calibration : 1;
                    } eltwise;
                    struct lstm_gemm_t {
                        uint32_t bias : 1;
                        uint32_t hidden : 1;
                    } lstm_gemm;
                    struct lstm_dynamic_t {
                        uint32_t last_hidden : 1;
                        uint32_t last_cell : 1;
                    } lstm_dynamic;
                    struct lstm_elt_t {
                        uint32_t cell : 1;
                    } lstm_elt;
                    struct fused_conv_eltw_t {
                        // conv
                        uint32_t split : 1;
                        uint32_t dilation : 1;
                        uint32_t depthwise_separable_opt : 1;
                        uint32_t transposed : 1;
                        uint32_t quantization : 1;
                        uint32_t calibration : 1;
                        uint32_t local : 1;
                        uint32_t grouped : 1;
                        // eltw
                        uint32_t stride : 1;
                        // fused conv eltw
                        uint32_t rw_out_opt : 1;
                    } fused_conv_eltw;
                    struct quantize_t {
                        uint32_t packed_binary_output : 1;
                        uint32_t scale_shift_opt : 1;
                    } quantize;
                } dedicated;
            } val;
            uint64_t raw;
        } restrict;

        union machine_info_t {
            struct val_t {
                uint32_t subgroup : 1;
                uint32_t subgroupShort : 1;
            } val;
            uint32_t raw;
        } machineInfo;

        static_assert(sizeof(restrict_t) == sizeof(uint64_t), "problem with union");

        typedef union DataTypesKey_t {
            struct val_t {
                uint32_t int8 : 1;
                uint32_t uint8 : 1;
                uint32_t int16 : 1;
                uint32_t uint16 : 1;
                uint32_t int32 : 1;
                uint32_t uint32 : 1;
                uint32_t int64 : 1;
                uint32_t F16 : 1;
                uint32_t F32 : 1;
                uint32_t binary : 1;
            } val;
            uint32_t raw;
        } DataTypesKey;

        uint32_t enableTuning;
        DataTypesKey inputType;
        DataTypesKey outputType;
        DataTypesKey inputWeightsType;
        DataTypesKey outputWeightsType;
        uint32_t inputLayout;
        uint32_t outputLayout;
        uint64_t weightsInputLayout;
        uint64_t weightsOutputLayout;

        static_assert(std::numeric_limits<decltype(weightsInputLayout)>::digits >= WeightsLayout::WeightsLayoutCount,
                      "Not enough bits in weightsInputLayout to store WeightLayout bitfield");

        static_assert(std::numeric_limits<decltype(weightsOutputLayout)>::digits >= WeightsLayout::WeightsLayoutCount,
                      "Not enough bits in weightsOutputLayout to store WeightLayout bitfield");

        static_assert(std::numeric_limits<decltype(inputLayout)>::digits >= DataLayout::DataLayoutCount,
                      "Not enough bits in inputLayout to store DataLayout bitfield");

        static_assert(std::numeric_limits<decltype(outputLayout)>::digits >= DataLayout::DataLayoutCount,
                      "Not enough bits in outputLayout to store DataLayout bitfield");
    };

    void EnableInputDataType(Datatype dt);
    void EnableAllInputDataType();
    void EnableOutputDataType(Datatype dt);
    void EnableAllOutputDataType();
    void EnableInputWeightsType(WeightsType wt);
    void EnableAllInputWeightsType();
    void EnableOutputWeightsType(WeightsType wt);
    void EnableAllOutputWeightsType();
    void EnableFP16Emulation() { key.restrict.val.FP16Emulation = 1; }
    void EnableDifferentTypes() { key.restrict.val.different_types = 1; }
    void EnableDifferentInputWeightsTypes() { key.restrict.val.different_input_weights_types = 1; }
    void EnableInputLayout(DataLayout l) { key.inputLayout |= (1 << l); }
    void EnableAllInputLayout() { key.inputLayout = ~static_cast<decltype(key.inputLayout)>(0); }
    void EnableOutputLayout(DataLayout l) { key.outputLayout |= (1 << l); }
    void EnableAllOutputLayout() { key.outputLayout = ~static_cast<decltype(key.outputLayout)>(0); }
    void EnableInputWeightsLayout(WeightsLayout l) { key.weightsInputLayout |= ((uint64_t)1 << l); }
    void EnableAllInputWeightsLayout() { key.weightsInputLayout = ~static_cast<decltype(key.weightsInputLayout)>(0); }
    void EnableOutputWeightsLayout(WeightsLayout l) { key.weightsOutputLayout |= ((uint64_t)1 << l); }
    void EnableAllOutputWeightsLayout() { key.weightsOutputLayout = ~static_cast<decltype(key.weightsOutputLayout)>(0); }
    void EnableTensorOffset() { key.restrict.val.offset = 1; }
    void EnableTensorPitches() { key.restrict.val.pitches = 1; }
    void EnableBatching() { key.restrict.val.batching = 1; }
    void EnableGradient() { key.restrict.val.gradient = 1; }
    void EnableSubGroup() { key.machineInfo.val.subgroup = 1; }
    void EnableSubGroupShort() { key.machineInfo.val.subgroupShort = 1; }
    void EnableNonBiasTerm() { key.restrict.val.nonBias = 1; }
    void EnableBiasPerFeature() { key.restrict.val.biasPerFeatureMap = 1; }
    void EnableBiasPerOutput() { key.restrict.val.biasPerOutput = 1; }
    void EnableActivationAdditionalParamsAsInput() { key.restrict.val.activationAdditionalParamsAsInput = 1; }
    void EnableMomentum() { key.restrict.val.momentum = 1; }
    void EnableLRNMode(LRNMode m);
    void EnableLookUpTableAxis(LookUpTableAxis m);
    void EnableNormalizeMode(NormalizeMode m);
    void EnableMVNMode(MVNMode m);
    void EnableMVNNormalizeVariance();
    void EnableLRNKernelDividerMode(KernelDividerMode m);
    void EnablePoolKernelDividerMode(KernelDividerMode m);
    void EnablePoolType(PoolType t);
    void EnablePoolRemainder(PoolRemainder r);
    void EnableQuantization(QuantizationType q);
    void EnablePositionSensitivePooling() { key.restrict.val.dedicated.pooling.position_sensitive = 1; }
    void EnableSplitSupport() { key.restrict.val.dedicated.conv.split = 1; }
    void EnableDilation() { key.restrict.val.dedicated.conv.dilation = 1; }
    void EnableDepthwiseSeparableOpt() { key.restrict.val.dedicated.conv.depthwise_separable_opt = 1; }
    void EnableLocalConvolution() { key.restrict.val.dedicated.conv.local = 1; }
    void EnableGroupedConvolution() { key.restrict.val.dedicated.conv.grouped = 1; }
    void EnableTranspose() { key.restrict.val.dedicated.conv.transposed = 1; }
    void EnableInt8Quantization() { key.restrict.val.quantization = 1; }
    void EnableOutputCalibration() { key.restrict.val.output_calibration = 1; }
    void EnableDeformableMode() { key.restrict.val.dedicated.conv.deformable = 1; }

    void EnableFusedConvEltwSplitSupport() { key.restrict.val.dedicated.fused_conv_eltw.split = 1; }
    void EnableFusedConvEltwDilation() { key.restrict.val.dedicated.fused_conv_eltw.dilation = 1; }
    void EnableFusedConvEltwDepthwiseSeparableOpt() {
        key.restrict.val.dedicated.fused_conv_eltw.depthwise_separable_opt = 1;
    }
    void EnableFusedConvEltwLocalConvolution() { key.restrict.val.dedicated.fused_conv_eltw.local = 1; }
    void EnableFusedConvEltwGroupedConvolution() { key.restrict.val.dedicated.fused_conv_eltw.grouped = 1; }
    void EnableFusedConvEltwTranspose() { key.restrict.val.dedicated.fused_conv_eltw.transposed = 1; }
    void EnableFusedConvEltwInt8Quantization() { key.restrict.val.dedicated.fused_conv_eltw.quantization = 1; }
    void EnableFusedConvEltwOutputCalibration() { key.restrict.val.dedicated.fused_conv_eltw.calibration = 1; }
    void EnableFusedConvEltwEltwiseStride();

    void EnableQuantizePackedBinaryOutput() { key.restrict.val.dedicated.quantize.packed_binary_output = 1; }
    void EnableQuantizeScaleShiftOpt() { key.restrict.val.dedicated.quantize.scale_shift_opt = 1; }

    void EnableWinogradReorder() { key.restrict.val.dedicated.reorder.winograd = 1; }
    void EnableSoftmaxDim(SoftmaxDim d);
    void EnableConcatAxis(ConcatAxis a);
    void EnableReampleType(ResampleType a);
    void EnableEltwiseStride();
    void EnableEltwiseBroadcast() { key.restrict.val.dedicated.eltwise.broadcast = 1; }
    void EnableEltwiseInputsCalibration() { key.restrict.val.dedicated.eltwise.inputs_calibration = 1; }

    void EnableLSTMGEMMBias() { key.restrict.val.dedicated.lstm_gemm.bias = 1; }
    void EnableLSTMGEMMHidden() { key.restrict.val.dedicated.lstm_gemm.hidden = 1; }
    void EnableLSTMEltCell() { key.restrict.val.dedicated.lstm_elt.cell = 1; }
    void EnableLSTMDyanmicOptionalHiddenOutput() { key.restrict.val.dedicated.lstm_dynamic.last_hidden = 1; }
    void EnableLSTMDyanmicOptionalCellOutput() { key.restrict.val.dedicated.lstm_dynamic.last_cell = 1; }
    void EnableConcatKernelPerInput() { key.restrict.val.dedicated.concat.kernelPerInput = 1; }
    void DisableTuning() { key.enableTuning = 0; }
    void EnableConcatOneKernel() { key.restrict.val.dedicated.concat.oneKernel = 1; }
    void EnableArgMaxMinAxis(ArgMaxMinAxis a);
    void EnableLookUpTableIndicesFormat(Datatype a);
    void EnableIndexSelectAxis(IndexSelectAxis a);
    void EnableFusedConvEltwiseRWOutOpt();
    bool Support(const ParamsKey& k) const;
    bool TuningSupport() const {
        if (key.enableTuning == 1)
            return true;
        return false;
    }
    bool isEnabledDifferentInputWeightsTypes() const {
        return key.restrict.val.different_input_weights_types ? true : false;
    }
    ParamsKey Merge(const ParamsKey& k) const;

private:
    Key key;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EngineInfo
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EngineInfo {
    bool bSubGroupSupport = false;
    bool bSubGroupShortSupport = false;
    bool bFP16Support = false;
    bool bFP64Support = false;
    bool bImageSupport = false;
    bool bIMADSupport = false;
    bool bIMMADSupport = false;
    uint32_t computeUnitsCount = 0;
    uint64_t maxWorkGroupSize = 0;
    uint64_t maxLocalMemSize = 0;
    uint64_t maxImage2dWidth = 0;
    uint64_t maxImage2dHeight = 0;
    std::string deviceId = "";
    std::string driverVersion = "";
    std::string hostVersion = "";
    std::shared_ptr<rapidjson::Document> deviceCache;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Params {
    virtual ~Params() {}

    KernelType GetType() const { return kType; }
    virtual ParamsKey GetParamsKey() const;

protected:
    Params(KernelType kt, const std::string& id) : kType(kt), layerID(id) {}
    KernelType kType;

public:
    std::string layerID;
    std::string forceImplementation;
    EngineInfo engineInfo;

    virtual std::string to_string() const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// base_activation_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct base_activation_params {
    ActivationFunction function = ActivationFunction::NONE;
    float m = 1.f;
    float n = 0.f;
    bool gradient = false;

    base_activation_params() = default;
    base_activation_params(const float m, const float n) : m(m), n(n) {}
    base_activation_params(const ActivationFunction f, const float m, const float n, const bool gradinet = false) : function(f),
                                                                                                                    m(m),
                                                                                                                    n(n),
                                                                                                                    gradient(gradinet) {}

    virtual std::string to_string() const;
};

struct FusedOpsConfiguration {
    enum class LoadType {
        LT_UNALIGNED = 0,
        LT_ALIGNED_READ = 1
    };

    enum class BoundaryCheck {
        DISABLED = 0,
        ENABLED = 1
    };

    enum class IndexType {
        TENSOR_COORD = 0,
        LINEAR_OFFSET = 1
    };

    std::string suffix;
    std::vector<std::string> bfzyx_idx_order;
    std::string input_var_name;
    Datatype input_dt;
    size_t vec_size;
    Tensor::DataChannelName vec_axis;
    LoadType load_type;
    BoundaryCheck boundary_check;
    IndexType index_type;

    FusedOpsConfiguration(std::string suffix,
                          std::vector<std::string> bfzyx_idx_order,
                          std::string input_var_name,
                          Datatype input_dt,
                          size_t vec_size = 1,
                          LoadType load_type = LoadType::LT_UNALIGNED,
                          BoundaryCheck boundary_check = BoundaryCheck::ENABLED,
                          IndexType index_type = IndexType::TENSOR_COORD,
                          Tensor::DataChannelName vec_axis = Tensor::DataChannelName::COUNT)
      : suffix(suffix)
      , bfzyx_idx_order(bfzyx_idx_order)
      , input_var_name(input_var_name)
      , input_dt(input_dt)
      , vec_size(vec_size)
      , vec_axis(vec_axis)
      , load_type(load_type)
      , boundary_check(boundary_check)
      , index_type(index_type) { }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// base_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct base_params : public Params {
    virtual ~base_params() {}

    // Instance of fused_operation_desc is added to fused_ops vector if a node has been fused to current one using program_impl::fuse_nodes
    // method. In order to process fused ops following modifications should be done in a kernel:
    // option 1 - using common generator:
    //     - create FusedOpsConfiguration object that contains configuration for common code generator.
    //       Multiple objects can be created if a kernel uses different data types at the same time. E.g. kernels that contains scalar and
    //       vector branches that are chosen in runtime. To handle this case, create 2 configurations with different suffixes, like
    //       "_SCALAR" and "_VEC" and then use generated macros accordingly.
    //     - add jit constants returned by KernelBase::MakeFusedOpsJitConstants method to the kernel's constants.
    //     - insert generated macros in the ocl code:
    //       in kernel declaration:
    //         #if HAS_FUSED_OPS_DECLS
    //           FUSED_OPS_DECLS,
    //         #endif
    //       in kernel body:
    //         #if HAS_FUSED_OPS
    //           FUSED_OPS<OPTIONAL_SUFFIX>;
    //           <SOME_VARIABLE> = FINAL_NAME<OPTIONAL_SUFFIX>;
    //         #endif
    //   In this case common generator creates set of definitions for each op which are called sequentially in FUSED_OP<OPTIONAL_SUFFIX>
    //   macro. Example:
    //     #define FUSED_OPS
    //       FUSED_OP0_LOAD_VEC
    //       FUSED_OP0_ACTION_VEC
    //       FUSED_OP1_LOAD_VEC
    //       FUSED_OP1_ACTION_VEC
    //     #define FUSED_OP0_LOAD_VEC
    //       MAKE_VECTOR_TYPE(FUSED_OP_0_INPUT0_TYPE,2) activation0_data0 = UNIT_BLOCK_READ(activation0_input0,
    //                                                                      FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(f_block*16),0,0));
    //     #define FUSED_OP0_ACTION_VEC
    //       float2 dst_0 = dst;
    //       dst_0 = ACTIVATION_FUSED_OP0_VEC(dst_0, ACTIVATION_PARAMS_FUSED_OP0_VEC);
    //     #define FUSED_OP1_LOAD_VEC
    //       MAKE_VECTOR_TYPE(FUSED_OP_1_INPUT0_TYPE,2) eltwise1_data0 = UNIT_BLOCK_READ2(eltwise1_input0,
    //                                                                   FUSED_OP_1_INPUT0_GET_INDEX_SAFE(0,(f_block*16),y,x));
    //     #define FUSED_OP1_ACTION_VEC
    //       float2 dst_0_2 = convert_float2(eltwise1_data0) + convert_float2(dst_0);
    //     #define FINAL_NAME_VEC dst_0_2
    // option 2 - using custom generator in a kernel. It can be used if performance is not optimal in the common one or to handle
    //            some difficult cases that can't be unified. Custom processing of fused ops can be written absolutely independently
    //            in a kernel, but to make it easier set of helper functions exist:
    //     - KernelBase::MakeFusedOpsDeclsJitConstants that creates arguments for kernel declaration and macro for all tensors used in
    //       a fused op (requires FusedOpsConfiguration instance).
    //     - fused_operation_desc contains a bunch of methods to generate variable/pointer names, type conversions, data loads
    //  If you need an example of custom code generation for fused ops, check BinaryConvolutionKernelGeneric::GetFusedPrimitivesJitConstants
    //  method in binary_convolution_kernel_generic.cpp.
    struct fused_operation_desc {
        struct idx_desc {
            std::string b;
            std::string f;
            std::string z;
            std::string y;
            std::string x;
            size_t dims;
            explicit idx_desc(std::vector<std::string> idx) : b("0"), f("0"), z("0"), y("0"), x("0"), dims(0) {
                dims = idx.size();
                switch (dims) {
                    case 1: f = idx[0]; break;
                    case 2: b = idx[0]; f = idx[1]; break;
                    case 3: b = idx[0]; f = idx[1]; y = idx[2]; break;
                    case 4: b = idx[0]; f = idx[1]; y = idx[2]; x = idx[3]; break;
                    case 5: b = idx[0]; f = idx[1]; z = idx[2]; y = idx[3]; x = idx[4]; break;
                    default: throw std::runtime_error("More than 5 dimenstions is not supported in fused op generator");
                }
            }
        };

        std::shared_ptr<fuse_params> op_params;
        size_t dep_idx_start;
        size_t dep_size;
        MultiDataTensor tensors;
        DataTensor output_tensor;
        size_t op_id;

        JitConstants MakeFusedTensorJitConstants(const FusedOpsConfiguration& conf) const;
        JitConstants MakeInputDeclsJitConstants(const FusedOpsConfiguration& conf) const;
        JitConstants MakeLoadJitConstants(const FusedOpsConfiguration& conf, const DataTensor prim_output) const;
        JitConstants MakeOpJitConstants(const FusedOpsConfiguration& conf,
                                        const std::string in_var, const Datatype in_type,
                                        std::string& out_var, Datatype& out_type) const;

        // Helper functions for operation generation
        KernelType GetType() const { return op_params->GetType(); }
        template<typename T>
        std::shared_ptr<T> GetOpParams() const {
            auto p = std::dynamic_pointer_cast<T>(op_params);
            if (!p)
                throw std::runtime_error("Invalid dynamic cast of fused operation parameters");

            return p;
        }
        std::string GetTypeStr() const;
        std::string GetInputTensorName(size_t input_id) const;
        std::string GetOutputTensorName() const;
        std::string GetInputTypeName(size_t input_id, size_t vec_size) const;
        std::string GetJitLoad(const FusedOpsConfiguration& conf, size_t input_id, const DataTensor prim_output,
                               bool reuse_index = false, std::string reused_idx = "") const;
        std::string GetIdx(size_t input_id, idx_desc idx, bool should_be_safe) const;
        std::string GetInputPtrName(size_t input_id) const;
        std::string GetInputVarName(size_t input_id) const;
        std::string GetOutputVarName(std::string input_var_name) const;
        std::string ConvertToOutputType(std::string var, size_t vec_size = 1) const;
        std::string ConvertToType(std::string var, Datatype dt, size_t vec_size = 1) const;
        std::string CastToType(std::string var, Datatype dt, size_t vec_size = 1) const;
        std::string Broadcast(std::string var,  Datatype dt, size_t vec_size = 1) const;
        std::string ConvertToOutputTypeSat(std::string var, size_t vec_size = 1) const;
        std::string GetOutputType(size_t vec_size = 1) const;
        std::string GetType(Datatype dt, size_t vec_size = 1) const;

    private:
        std::vector<size_t> GetRequiredInputs() const;
    };

    std::vector<base_activation_params> activations;
    std::vector<fused_operation_desc> fused_ops = {};
    MultiDataTensor inputs;
    DataTensor output;
    bool gradient = false;

    virtual std::string to_string() const;
    virtual ParamsKey GetParamsKey() const;

protected:
    explicit base_params(KernelType kt) : Params(kt, ""), inputs(1) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Auto tuner parameters
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class KernelRunnerInterface;
struct TuningParams {
    TuningMode mode;
    std::string cacheFilePath;
    std::shared_ptr<KernelRunnerInterface> runner;

    TuningParams() : mode(TuningMode::TUNING_DISABLED), cacheFilePath(""), runner(nullptr) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct optional_params {
    virtual ~optional_params() {}

    KernelType GetType() const { return kType; }

    std::vector<DataLayout> inputLayouts;
    std::vector<DataLayout> outputLayouts;

    bool meaningfulKernelsNames = false;  // use layer name instead of internal kernel name
    bool allowStaticInputReordering =
        true;  // allow kernel to provide a kernel which reorder static data like weights/bias/tables...
    bool allowInputReordering =
        false;  // allow kernel to ask graph compiler to reorder the input data before executing its
    bool allowOutputReordering =
        false;  // allow kernel to ask graph compiler to reorder the output data before executing the next kernel

    TuningParams tuningParams;

    virtual ParamsKey GetSupportedKey() const;

protected:
    explicit optional_params(KernelType kt) : kType(kt) {}
    KernelType kType;
};

}  // namespace kernel_selector
