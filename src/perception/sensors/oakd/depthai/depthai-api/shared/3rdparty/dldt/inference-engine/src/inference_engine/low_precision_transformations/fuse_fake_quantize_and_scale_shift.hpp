// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <ie_common.h>
#include <algorithm>
#include "low_precision_transformations/fake_quantize.hpp"

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(FuseFakeQuantizeAndScaleShiftTransformation) : public FakeQuantizeTransformation {
public:
    FuseFakeQuantizeAndScaleShiftTransformation(const Params& params) : FakeQuantizeTransformation(params) {}
    ~FuseFakeQuantizeAndScaleShiftTransformation() override {};

    void transform(TransformationContext& context, CNNLayer& layer) const override;
};

}  // namespace details
}  // namespace InferenceEngine
