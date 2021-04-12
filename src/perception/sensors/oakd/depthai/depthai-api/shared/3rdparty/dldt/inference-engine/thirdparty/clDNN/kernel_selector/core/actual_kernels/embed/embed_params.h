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

#include "weight_bias_params.h"
#include <string>

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// embed_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct embed_params : public weight_bias_params {
    embed_params() : weight_bias_params(KernelType::EMBED) {}

    std::string to_string() const {
        std::stringstream s;

        s << base_params::to_string() << "_";
        if (bias.empty()) {
            s << "no_bias"
              << "_";
        } else {
            s << "bias_" << bias[0].PhysicalSize() << "_";
        }
        return s.str();
    }
    virtual ParamsKey GetParamsKey() const { return weight_bias_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// embed_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct embed_optional_params : weight_bias_optional_params {
    embed_optional_params() : weight_bias_optional_params(KernelType::EMBED) {}
};
}  // namespace kernel_selector