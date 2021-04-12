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

#include "weight_bias_params.h"
#include <string>
#include <vector>

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct convolution_params : public weight_bias_params {
    convolution_params() : weight_bias_params(KernelType::CONVOLUTION) {}
    uSize filterSize;
    uSize stride;
    uSize dilation;
    uSize padding;
    uint32_t split = 1;
    bool depthwise_separable_opt = false;
    bool transposed = false;
    QuantizationType quantization = QuantizationType::NONE;
    bool has_compensation = false;
    bool local_convolution = false;
    bool deformable_mode = false;
    uint32_t groups = 1;
    uSize kernelSize;
    uint32_t deformable_groups = 1;

    MultiDataTensor weights_zero_points;
    MultiDataTensor activations_zero_points;
    MultiDataTensor compenstaion;
    std::string to_string() const override;
    ParamsKey GetParamsKey() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct convolution_optional_params : weight_bias_optional_params {
    convolution_optional_params() : weight_bias_optional_params(KernelType::CONVOLUTION) {}
};

}  // namespace kernel_selector
