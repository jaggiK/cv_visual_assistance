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


#include "convolution_grad_weights_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>
#include <algorithm>

namespace kernel_selector {
std::string convolution_grad_weights_params::to_string() const {
    std::stringstream s;

    s << base_params::to_string() << "_";
    if (bias.empty()) {
        s << "no_bias"
          << "_";
    } else {
        s << "bias_" << bias[0].PhysicalSize() << "_";
    }
    s << filterSize.x << "_" << filterSize.y << "_";
    s << stride.x << "_" << stride.y << "_";
    s << dilation.x << "_" << dilation.y << "_";
    s << padding.x << "_" << padding.y << "_";
    s << split;

    return s.str();
}

JitConstants ConvolutionGradWeightsKernelBase::GetJitConstants(const convolution_grad_weights_params& cp) const {
    JitConstants jit = training_kernel_base::GetJitConstants(cp);
    const auto& padding = cp.padding;
    const auto& input = cp.inputs[0];

    int64_t input_offset_with_padding = (int64_t)input.GetFirstElementOffset() -
                                        (cp.filterSize.x - 1 + padding.x) * input.X().pitch -
                                        (cp.filterSize.y - 1 + padding.y) * input.Y().pitch;
    input_offset_with_padding = std::max(input_offset_with_padding, (int64_t)0);

    jit.AddConstants({
        MakeJitConstant("STRIDE", cp.stride),
        MakeJitConstant("PADDING", cp.padding),
        MakeJitConstant("DILATION", cp.dilation),
        MakeJitConstant("FILTER_ARRAY_NUM", cp.split),
        MakeJitConstant("INPUT0_OFFSET_WITH_PADDING", input_offset_with_padding),
        MakeJitConstant("DEPTHWISE_SEPARABLE_OPT", cp.depthwise_separable_opt),
        MakeJitConstant("OUTPUT_GRAD_W", cp.output_grad_w),
    });

    return jit;
}

ConvolutionGradWeightsKernelBase::DispatchData ConvolutionGradWeightsKernelBase::SetDefault(
    const convolution_grad_weights_params& params) const {
    auto input_features = params.weights.IFM().v;
    auto output_features = params.weights.OFM().v;

    DispatchData kd;

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
    size_t gws0 = output_features * input_features;
    size_t lws0 = std::min(gws0, static_cast<size_t>(32));
    while (gws0 % lws0) {
        lws0--;
    }
    kd.gws0 = gws0;
    kd.gws1 = params.weights.X().v;
    kd.gws2 = params.weights.Y().v;
    kd.lws0 = lws0;
    kd.lws1 = 1;
    kd.lws2 = 1;
    kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
    return kd;
}

KernelsData ConvolutionGradWeightsKernelBase::GetKernelsData(const Params& params,
                                                             const optional_params& options) const {
    assert(params.GetType() == KernelType::CONVOLUTION_GRAD_WEIGHTS);

    if (!Validate(params, options)) {
        return {};
    }

    const convolution_grad_weights_params& orgParams = static_cast<const convolution_grad_weights_params&>(params);

    const std::vector<WeightsLayout> weightsLayouts = {WeightsLayout::oiyx,
                                                       WeightsLayout::iyxo,
                                                       WeightsLayout::yxio,
                                                       WeightsLayout::oyxi};

    DispatchData runInfo = SetDefault(orgParams);
    KernelData kd = KernelData::Default<convolution_grad_weights_params>(params);
    convolution_grad_weights_params& newParams = *static_cast<convolution_grad_weights_params*>(kd.params.get());

    bool succeed = UpdateWeightsParams(newParams, options, weightsLayouts, kd.weightsReorderParams);

    if (!succeed) {
        return {};
    }

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     runInfo,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     true,
                     !orgParams.bias.empty());
    if (newParams.use_momentum) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::PREV_WEIGHTS_GRADIENT, 0});
        if (!newParams.bias.empty())
            kernel.arguments.push_back({ArgumentDescriptor::Types::PREV_BIAS_GRADIENT, 0});
    }
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    kernel.arguments.push_back({ArgumentDescriptor::Types::SPLIT, 0});
    kernel.arguments.push_back({ArgumentDescriptor::Types::LEARNING_RATE, 0});

    kd.estimatedTime = runInfo.effiency;

    return {kd};
}
}  // namespace kernel_selector