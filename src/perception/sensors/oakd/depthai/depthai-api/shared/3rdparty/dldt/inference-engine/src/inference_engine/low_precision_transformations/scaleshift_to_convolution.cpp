﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <details/caseless.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

static const std::unordered_set<std::string> defaultIgnoreWithParents = {
    "Convolution",
    "FakeQuantize"
};

ScaleShiftToConvolutionTransformation::ScaleShiftToConvolutionTransformation(const Params& params) :
    WeightableLayerTransformation(params),
    groupSize(1ul),
    ignoreWithParents(defaultIgnoreWithParents) {
}

void ScaleShiftToConvolutionTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!CaselessEq<std::string>()(layer.type, "ScaleShift")) {
        THROW_IE_EXCEPTION << "Layer '" << layer.name << "' has invalid type '" << layer.type << "'. Convolution is expected.";
    }

    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
    if (parents.size() != 1)
        return;
    const DataPtr outData = CNNNetworkHelper::getOutData(*parents[0], layer);
    if (outData == nullptr) {
        THROW_IE_EXCEPTION << "layer " << layer.type << " '" << layer.name << "' is child for " << parents[0]->type << " '" << parents[0]->name << "'";
    }

    if (outData->getInputTo().size() == 1ul && parents[0]->type != "Concat")
        return;

    if (layer.outData[0]->getInputTo().size() == 0ul) {
        return;
    }

    if (updatePrecisions) {
        const Precision parentPrecision = CNNNetworkHelper::getPrecisionParent(layer);
        if ((parentPrecision != Precision::I8) && (parentPrecision != Precision::U8)) {
            return;
        }
    }

    if (std::any_of(parents.begin(), parents.end(), [](CNNLayerPtr parent) { return CaselessEq<std::string>()(parent->type, "Input"); })) {
        return;
    }

    const size_t channelsCount = CNNNetworkHelper::getOutputChannelsCount(layer);
    if (channelsCount != CNNNetworkHelper::getInputChannelsCount(layer)) {
        return;
    }

    if (channelsCount % groupSize != 0) {
        return;
    }

    if (layer.insData[0].lock()->getDims().size() != 4) {
        return;
    }

    CNNLayerPtr convolutionLayerPtr = transformToConvolution(
        context,
        layer,
        channelsCount / groupSize);

    if (updatePrecisions) {
        std::vector<float> originalDataDequantizationScales(channelsCount, 1.f);
        std::vector<float> originalDataDequantizationShifts(channelsCount, 0.f);
        std::vector<float> originalWeightsDequantizationScales(channelsCount);
        const Blob::Ptr weightsOriginalShiftsBlob = CNNNetworkHelper::getBlob(std::make_shared<CNNLayer>(layer), "weights");
        const float* weightsOriginalShiftsBuffer = weightsOriginalShiftsBlob->buffer().as<float*>();
        for (size_t i = 0ul; i < originalWeightsDequantizationScales.size(); ++i) {
            originalWeightsDequantizationScales[i] = weightsOriginalShiftsBuffer[i];
        }
        std::vector<float> originalWeightsDequantizationShifts(channelsCount, 0.f);
        std::vector<float> dequantizationScales;
        std::vector<float> dequantizationShifts;
        calculateDequantizationForSymmetric(
            *convolutionLayerPtr,
            originalDataDequantizationScales,
            originalDataDequantizationShifts,
            originalWeightsDequantizationScales,
            originalWeightsDequantizationShifts,
            dequantizationScales,
            dequantizationShifts);

        if (this->updateBiases) {
            std::vector<float> biasesShifts(dequantizationShifts.size(), 0.f);
            updateLayerBiases(context, *convolutionLayerPtr, dequantizationScales, dequantizationShifts, biasesShifts);
        }

        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*convolutionLayerPtr);
        if (children.size() == 0) {
            const std::string originalName = convolutionLayerPtr->name;
            CNNNetworkHelper::renameLayer(context.network, convolutionLayerPtr->name, convolutionLayerPtr->name + LayerTransformation::lastLayerPrefix);

            const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                convolutionLayerPtr,
                nullptr,
                DequantizationDetails(dequantizationScales, dequantizationShifts, channelsCount),
                originalName);
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        } else {
            for (const CNNLayerPtr& child : children) {
                const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context,
                    convolutionLayerPtr,
                    child,
                    DequantizationDetails(dequantizationScales, dequantizationShifts, channelsCount));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }
    }
}

void ScaleShiftToConvolutionTransformation::setGroupSize(const size_t groupSize) {
    this->groupSize = groupSize;
}

size_t ScaleShiftToConvolutionTransformation::getGroupSize() const {
    return groupSize;
}

void ScaleShiftToConvolutionTransformation::setIgnoreWithParents(const std::unordered_set<std::string>& ignoreWithParents) {
    this->ignoreWithParents = ignoreWithParents;
}

std::unordered_set<std::string> ScaleShiftToConvolutionTransformation::getIgnoreWithParents() const {
    return ignoreWithParents;
}

bool ScaleShiftToConvolutionTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return false;
}

bool ScaleShiftToConvolutionTransformation::isQuantized(const CNNLayer& layer) const noexcept {
    return true;
}

CNNLayerPtr ScaleShiftToConvolutionTransformation::transformToConvolution(
    TransformationContext& context,
    const CNNLayer& layer,
    const size_t group) const {
    const Precision originalPrecision = layer.outData[0]->getTensorDesc().getPrecision();
    const LayerParams convolutionLayerParams{ layer.name, "Convolution", originalPrecision };
    CNNLayerPtr convolutionLayerPtr = std::make_shared<ConvolutionLayer>(convolutionLayerParams);
    ConvolutionLayer* convolutionLayer = dynamic_cast<ConvolutionLayer*>(convolutionLayerPtr.get());
    convolutionLayer->_kernel.insert(X_AXIS, 1);
    convolutionLayer->_kernel.insert(Y_AXIS, 1);
    convolutionLayer->params["kernel"] = "1,1";
    convolutionLayer->_stride.insert(X_AXIS, 1);
    convolutionLayer->_stride.insert(Y_AXIS, 1);
    convolutionLayer->_padding.insert(X_AXIS, 0);
    convolutionLayer->_padding.insert(Y_AXIS, 0);
    convolutionLayer->_pads_end.insert(X_AXIS, 0);
    convolutionLayer->_pads_end.insert(Y_AXIS, 0);
    convolutionLayer->_dilation.insert(X_AXIS, 1);
    convolutionLayer->_dilation.insert(Y_AXIS, 1);
    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(layer);
    convolutionLayer->_out_depth = outputChannelsCount;
    convolutionLayer->_group = group;
    convolutionLayer->params["group"] = std::to_string(group);

    CNNLayerPtr layerPtr = std::make_shared<CNNLayer>(layer);
    CNNNetworkHelper::replaceLayer(context, layerPtr, convolutionLayerPtr);

    {
        const Precision weightsPrecision = updatePrecisions ? precisionsOnWeights[0] : CNNNetworkHelper::getPrecisionParent(layer);
        const Precision biasesPrecision = originalPrecision;

        LayerParams weightsLayerParams{ layer.name + "Weights", "Const", weightsPrecision };
        CNNLayerPtr weightsConstLayer = std::make_shared<CNNLayer>(weightsLayerParams);
        CNNNetworkHelper::addLayer(context, nullptr, convolutionLayerPtr, weightsConstLayer);

        {
            const size_t inputChannelsCount = CNNNetworkHelper::getInputChannelsCount(layer);
            const size_t weightsSize = outputChannelsCount * inputChannelsCount / group;
            std::shared_ptr<float> weightsBufferPtr(new float[weightsSize], std::default_delete<float[]>());
            float* weightsBuffer = weightsBufferPtr.get();

            const Blob::Ptr weightsOriginalShiftsBlob = CNNNetworkHelper::getBlob(std::make_shared<CNNLayer>(layer), "weights");
            const float* weightsOriginalShiftsBlobBuffer = weightsOriginalShiftsBlob->buffer().as<float*>();
            const size_t kernelsCount = inputChannelsCount / group;
            if (group == 1ul) {
                for (size_t outputChannel = 0ul; outputChannel < outputChannelsCount; ++outputChannel) {
                    for (size_t kernel = 0ul; kernel < kernelsCount; ++kernel) {
                        const float value = (outputChannel == kernel) ? (updatePrecisions ? 1.f : weightsOriginalShiftsBlobBuffer[outputChannel]) : 0.f;
                        weightsBuffer[kernelsCount * outputChannel + kernel] = value;
                    }
                }
            } else {
                const float channelsInGroup = outputChannelsCount / group;
                for (size_t outputChannel = 0ul; outputChannel < outputChannelsCount; ++outputChannel) {
                    const size_t groupIndex = outputChannel / channelsInGroup;
                    for (size_t kernel = 0ul; kernel < kernelsCount; ++kernel) {
                        const size_t outputChannelIndexInGroup = outputChannel - groupIndex * channelsInGroup;
                        const float value = (outputChannelIndexInGroup == kernel) ?
                            (updatePrecisions ? 1.f : weightsOriginalShiftsBlobBuffer[outputChannel]) : 0.f;
                        weightsBuffer[kernelsCount * outputChannel + kernel] = value;
                    }
                }
            }

            Blob::Ptr weights = CNNNetworkHelper::makeNewBlobPtr(TensorDesc(weightsPrecision, { weightsSize }, Layout::C));
            weights->allocate();
            CNNNetworkHelper::fillBlobByFP32(weights, weightsBuffer);
            weightsConstLayer->blobs["custom"] = weights;
            weightsConstLayer->outData[0]->reshape({ outputChannelsCount, inputChannelsCount / group, 1, 1 }, Layout::NCHW);
            weightsConstLayer->outData[0]->setPrecision(weightsPrecision);
            // TODO: workaround
            weightsConstLayer->precision = weightsPrecision;
        }

        LayerParams biasesLayerParams{ layer.name + "Biases", "Const", biasesPrecision };
        CNNLayerPtr biasesConstLayer = std::make_shared<CNNLayer>(biasesLayerParams);
        CNNNetworkHelper::addLayer(context, nullptr, convolutionLayerPtr, biasesConstLayer);

        Blob::Ptr biasesOriginalShiftsBlob = CNNNetworkHelper::getBlob(std::make_shared<CNNLayer>(layer), "biases");
        biasesConstLayer->blobs["custom"] = biasesOriginalShiftsBlob;
        biasesConstLayer->outData[0]->reshape({ biasesOriginalShiftsBlob->size() }, Layout::C);
    }

    return convolutionLayerPtr;
}
