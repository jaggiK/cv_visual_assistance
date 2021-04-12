// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {
namespace details {

/**
* @brief Quantization layer details and basic operations on them.
*/
class INFERENCE_ENGINE_API_CLASS(QuantizationDetails) {
public:
    QuantizationDetails();
    QuantizationDetails(const QuantizationDetails& quantizationDetails);
    QuantizationDetails(
        const size_t levels,
        const std::vector<float>& inputLowValues,
        const std::vector<float>& inputHighValues,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues,
        const size_t inputIntervalsCount,
        const size_t outputIntervalsCount,
        const size_t outputChannelsCount);

    static QuantizationDetails getDetails(const CNNLayer& quantize);
    bool hasNegativeOutput() const;
    float maxOutput(const size_t channel) const;
    float maxInput(const size_t channel) const;

    float maxOutputHigh() const;
    float minOutputLow() const;

    float getInputLowValue(const size_t channel) const;
    float getInputHighValue(const size_t channel) const;
    float getOutputLowValue(const size_t channel) const;
    float getOutputHighValue(const size_t channel) const;

    static bool isSupportedLevel(const size_t level);

    const size_t levels;
    const std::vector<float> inputLowValues;
    const std::vector<float> inputHighValues;
    const std::vector<float> outputLowValues;
    const std::vector<float> outputHighValues;
    const size_t inputIntervalsCount;
    const size_t outputIntervalsCount;
    const size_t outputChannelsCount;

private:
    QuantizationDetails &operator=(const QuantizationDetails &terget) { return *this; }
    static void validate(const CNNLayerPtr& constantLayer);
    static std::vector<float> getBlobValue(const CNNLayerPtr& constantLayer);
};

inline std::ostream &operator << (std::ostream &os, const QuantizationDetails& value) {
    os << "levels: " << value.levels <<
        ", input 1/" << value.inputIntervalsCount << ": [" << value.getInputLowValue(0) << " : " << value.getInputHighValue(0) << "], " <<
        ", output 1/" << value.outputIntervalsCount << ": [" << value.getOutputLowValue(0) << " : " << value.getOutputHighValue(0) << "]";
    return os;
}

}  // namespace details
}  // namespace InferenceEngine
