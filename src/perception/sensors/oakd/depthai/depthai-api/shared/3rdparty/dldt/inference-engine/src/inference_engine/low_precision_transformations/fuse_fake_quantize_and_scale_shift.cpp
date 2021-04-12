﻿// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_fake_quantize_and_scale_shift.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <details/caseless.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void FuseFakeQuantizeAndScaleShiftTransformation::transform(
        TransformationContext& context,
        CNNLayer& fakeQuantizeLayer) const {
    if (!CaselessEq<std::string>()(fakeQuantizeLayer.type, "FakeQuantize"))
        return;

    // Fuse if only all children are ScaleShift
    auto dScaleShiftsVector = CNNNetworkHelper::getChildren(fakeQuantizeLayer);
    for (const auto& child : dScaleShiftsVector) {
        if (!CaselessEq<std::string>()(child->type, "ScaleShift"))
            return;

        const DataPtr insData = child->insData[0].lock();
        if (insData == nullptr) {
            return;
        }

        if (insData->getDims().size() > 5) {
            return;
        }
    }

    auto dScaleShift = dScaleShiftsVector[0];
    const Blob::Ptr scalesBlob = CNNNetworkHelper::getBlob(dScaleShift, "weights");
    auto scalesBufferPtr = CNNNetworkHelper::getFloatData(scalesBlob);

    const Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(dScaleShift, "biases");
    auto shiftsBufferPtr = CNNNetworkHelper::getFloatData(shiftsBlob);

    if (scalesBlob->size() != shiftsBlob->size())
        THROW_IE_EXCEPTION << "Scales and shifts values count are different for layer '" << dScaleShift->name << "'";

    const float* shiftsBuffer = shiftsBufferPtr.get();
    const float* scalesBuffer = scalesBufferPtr.get();
    // Don't fuse when there is a negative scale, because it leads to invalid results of FQ
    for (size_t i = 0lu; i < scalesBlob->size(); ++i) {
        if (scalesBuffer[i] <= 0.0f)
            return;
    }

    // All ScaleShifts must be equal
    for (size_t i = 1lu; i < dScaleShiftsVector.size(); i++) {
        auto ssLayer = dScaleShiftsVector[i];

        const Blob::Ptr scBlob = CNNNetworkHelper::getBlob(ssLayer, "weights");
        auto scBufferPtr = CNNNetworkHelper::getFloatData(scBlob);

        const Blob::Ptr shBlob = CNNNetworkHelper::getBlob(ssLayer, "biases");
        auto shBufferPtr = CNNNetworkHelper::getFloatData(shBlob);

        for (size_t j = 0lu; j < scalesBlob->size(); j++) {
            if (scalesBuffer[j] != scBufferPtr.get()[j] ||
                    shiftsBuffer[j] != shBufferPtr.get()[j])
                return;
        }
    }

    CNNLayerPtr outputLow = CNNNetworkHelper::getParent(fakeQuantizeLayer, 3);
    CNNLayerPtr outputHigh = CNNNetworkHelper::getParent(fakeQuantizeLayer, 4);

    const size_t inputDims = dScaleShift->insData[0].lock()->getDims().size();
    Layout layout;
    size_t channelIndex;
    switch (inputDims) {
        case 5: {
            layout = Layout::NCDHW;
            channelIndex = 1ul;
            break;
        }
        case 4: {
            layout = Layout::NCHW;
            channelIndex = 1ul;
            break;
        }
        case 3: {
            layout = Layout::BLOCKED;
            channelIndex = 1ul;
            break;
        }
        case 2: {
            layout = Layout::NC;
            channelIndex = 1ul;
            break;
        }
        case 1: {
            layout = Layout::C;
            channelIndex = 0ul;
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "FuseFakeQuantizeAndScaleShiftTransformation: unexpected dimensions count " <<
                inputDims << " in ScaleShift optimization";
        }
    }
    std::vector<size_t> dims(inputDims, 1lu);
    dims[channelIndex] = scalesBlob->size();

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantizeLayer);

    Blob::Ptr targetOutputLowBufferPtr = reshapeWeightsIntervalConst(*outputLow, dims, layout);
    auto targetOutputLowBuffer = CNNNetworkHelper::getFloatData(targetOutputLowBufferPtr);
    Blob::Ptr targetOutputHighBufferPtr = reshapeWeightsIntervalConst(*outputHigh, dims, layout);
    auto targetOutputHighBuffer = CNNNetworkHelper::getFloatData(targetOutputHighBufferPtr);

    for (size_t i = 0lu; i < scalesBlob->size(); ++i) {
        auto q_lo = quantizationDetails.getOutputLowValue(i);
        auto q_ho = quantizationDetails.getOutputHighValue(i);
        auto sc = scalesBlob->size() == 1lu ? scalesBuffer[0] : scalesBuffer[i];
        auto sh = shiftsBlob->size() == 1lu ? shiftsBuffer[0] : shiftsBuffer[i];
        targetOutputLowBuffer.get()[i] = q_lo * sc + sh;
        targetOutputHighBuffer.get()[i] = q_ho * sc + sh;
    }

    CNNNetworkHelper::fillBlobByFP32(targetOutputLowBufferPtr, targetOutputLowBuffer.get());
    CNNNetworkHelper::fillBlobByFP32(targetOutputHighBufferPtr, targetOutputHighBuffer.get());

    for (auto& ss : dScaleShiftsVector) {
        CNNNetworkHelper::removeLayer(context.network, ss);
        context.removeLayer(*ss);
    }
    if (updatePrecisions) {
        auto ssPrecision = dScaleShiftsVector[0]->outData[0]->getPrecision();
        fakeQuantizeLayer.outData[0]->setPrecision(ssPrecision);
    }
}
