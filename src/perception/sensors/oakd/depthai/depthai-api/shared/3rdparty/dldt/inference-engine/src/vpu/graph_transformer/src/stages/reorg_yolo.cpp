// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class ReorgYoloStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ReorgYoloStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto input = inputEdge(0)->input();

        auto inOrder = input->desc().dimsOrder();

        if (inOrder.dimInd(Dim::C) == 0) {
            stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
            stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto stride = attrs().get<int>("stride");

        serializer.append(static_cast<int32_t>(stride));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseReorgYolo(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    const auto maxC = 96;
    const auto maxH = 48;
    const auto maxW = 48;

    auto desc = inputs[0]->desc();
    const auto dimC = desc.dim(Dim::C);
    const auto dimH = desc.dim(Dim::H);
    const auto dimW = desc.dim(Dim::W);

    VPU_THROW_UNLESS((dimC <= maxC) && (dimH <= maxH) && (dimW <= maxW),
                     "ReorgYolo: too big tensor sizes to process: CHW %v %v %v, limits are: %v %v %v",
                     dimC, dimH, dimW, maxC, maxH, maxW);

    auto stage = model->addNewStage<ReorgYoloStage>(
        layer->name,
        StageType::ReorgYolo,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int>("stride", layer->GetParamAsInt("stride", 2));
}

}  // namespace vpu
