// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_layers.h>

#include <vpu/model/model.hpp>

namespace vpu {

class StageBuilder final {
public:
    using Ptr = std::shared_ptr<StageBuilder>;

    Stage createConvertStage(
            const Model& model,
            const std::string& name,
            const Data& input,
            const Data& output,
            float scale = 1.0f,
            float bias = 0.0f);

    Stage addSumStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input0,
            const Data& input1,
            const Data& output);

    Stage addBiasStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& biases,
            const Data& output);

    Stage addScaleStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& scales,
            const Data& output);

    Stage addCopyStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            const std::string& origin);

    Stage addPadStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            PadMode padMode,
            float pad_value,
            const DimValues& pads_begin,
            const DimValues& pads_end,
            const Data& input,
            const Data& output);

    Stage addNoneStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const DataVector& inputs,
            const DataVector& outputs);

    Stage addPowerStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float scale,
            float power,
            float bias,
            const Data& input,
            const Data& output);

    Stage addReLUStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float negativeSlope,
            const Data& input,
            const Data& output,
            const Data& biases = nullptr);

    Stage addReshapeStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output);

    Stage addConcatStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            Dim axis,
            const DataVector& inputs,
            const Data& output);

    Stage addConcatStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            std::vector<DimValues>&& offsets,
            const DataVector& inputs,
            const Data& output);

    Stage addSplitStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            Dim axis,
            const Data& input,
            const DataVector& outputs);

    Stage addSplitStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            std::vector<DimValues>&& offsets,
            const Data& input,
            const DataVector& outputs);

    Stage addScalingStage(
            const Model& model,
            const ie::CNNLayerPtr& origLayer,
            float scale,
            const Data& input,
            const Data& output);

    Stage addSwFullyConnectedStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& weights,
            const Data& biases,
            const Data& scales,
            Data output);

    Stage addExpandStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            const DimValues& offset = DimValues());

    Stage addShrinkStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            const DimValues& offset = DimValues());

    Stage addSoftMaxStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            Dim axis);

    Stage addClampStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float min,
            float max,
            const Data& input,
            const Data& output);

    Stage addGemmStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float alpha,
            float beta,
            bool transposeA,
            bool transposeB,
            const DataVector& inputs,
            const Data& output);


    Stage addGatherStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input0,
            const Data& input1,
            const Data& output,
            Dim axis);

    Stage addPermuteStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output,
            const DimValues_<Dim>& permutation);

    Stage addSCReluStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            float negativeSlope,
            Dim axis,
            const Data& input,
            const Data& output,
            const Data& scales,
            const Data& biases);

    Stage addReorderStage(
            const Model& model,
            const std::string& name,
            const ie::CNNLayerPtr& layer,
            const Data& input,
            const Data& output);

    Stage addReduceStage(
            const Model& model,
            const std::string& name,
            StageType reduceType,
            const ie::CNNLayerPtr& layer,
            bool keep_dims,
            const DataVector& inputs,
            const Data& output);

    Stage addLoopStartStage(
        const Model& model,
        const std::string& name,
        const DataVector& inputs,
        const DataVector& outputs);

    Stage addLoopEndStage(
        const Model& model,
        const std::string& name,
        const DataVector& inputs,
        const DataVector& outputs);
};

}  // namespace vpu
