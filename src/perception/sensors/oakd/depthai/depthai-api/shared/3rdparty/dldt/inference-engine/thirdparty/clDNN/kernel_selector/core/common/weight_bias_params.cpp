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

#include "weight_bias_params.h"

namespace kernel_selector {
ParamsKey weight_bias_params::GetParamsKey() const {
    ParamsKey k = base_params::GetParamsKey();

    k.EnableInputWeightsType(weights.GetDType());

    // not needed - can be changed by reorder params
    // k.EnableWeightsLayout(weights.layout);

    assert(bias.size() <= 1);

    if (bias.empty()) {
        k.EnableNonBiasTerm();
    } else if (bias[0].GetLayout() == DataLayout::bf || bias[0].GetLayout() == DataLayout::fb) {
        k.EnableBiasPerFeature();
    } else if (bias[0].GetLayout() == output.GetLayout()) {
        k.EnableBiasPerOutput();
    }

    return k;
}
}  // namespace kernel_selector