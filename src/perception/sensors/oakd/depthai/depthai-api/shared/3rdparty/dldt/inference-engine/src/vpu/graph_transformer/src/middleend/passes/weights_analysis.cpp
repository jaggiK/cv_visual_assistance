// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <cmath>

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <limits>
#include <memory>
#include <list>
#include <set>

#include <precision_utils.h>

#include <vpu/utils/numeric.hpp>
#include <vpu/compile_env.hpp>

#include <details/caseless.hpp>

namespace vpu {

namespace {

const short largestExp   = 15;
const short smallestExp  = -15;
const short exponentBias = 15;

short calculateExponent(short src) {
    short exponent;
    const short numberBitsFP16Mantiss = 10;
    const short numberBitsFP16Exponent = 5;
    src = src >> numberBitsFP16Mantiss;

    exponent = (src & ((1 << numberBitsFP16Exponent)-1));
    exponent -= exponentBias;
    return exponent;
}

std::vector<short> calculateExponents(const fp16_t* srcPtr, int count) {
    std::vector<short> exponents(count);

    for (int i = 0; i < count; ++i) {
        exponents[i] = calculateExponent(srcPtr[i]);
    }

    return exponents;
}

int getMeanValue(const std::vector<short>& exponents) {
    double sum = 0;
    int realSize = 0;
    for (int i = 0; i < exponents.size(); i++) {
        if (exponents[i] != smallestExp) {
            sum += exponents[i];
            realSize++;
        }
    }

    if (realSize == 0) {
        return smallestExp;
    } else {
        return sum / realSize;
    }
}

bool isScalable(const Stage& stage) {
    if (stage->type() != StageType::StubConv &&
        stage->type() != StageType::StubFullyConnected &&
        stage->type() != StageType::StubDeconv) {
        return false;
    }

    auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
    if (!tryHW) {
        return false;
    }

    return true;
}

bool checkGrowingOutput(const Model& model) {
    static const float SCALE_THRESHOLD = 0.125f;

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::Power &&
            stage->type() != StageType::Convert &&
            stage->type() != StageType::ScaleShift) {
            continue;
        }

        //
        // Power or ScaleShift stage may be located not only immediately
        // after Input. Convert layer can be located between Input and
        // Power/ScaleShift. Anyway, we need scale values from such
        // Power/ScaleShift stage in this pass.
        //

        auto input = stage->input(0);
        const auto& producer = input->producer();
        if (producer != nullptr && producer->type() == StageType::Convert) {
            input = producer->input(0);
        }

        if (input->usage() != DataUsage::Input) {
            continue;
        }

        if (stage->type() == StageType::Convert ||
            stage->type() == StageType::Power) {
            const auto scale = stage->attrs().get<float>("scale");
            if (scale < SCALE_THRESHOLD) {
                return false;
            }
        } else if (stage->type() == StageType::ScaleShift) {
            const auto scales = stage->input(1);

            const auto scalesContent = scales->content();
            IE_ASSERT(scalesContent != nullptr);

            const auto scalesContentPtr = scalesContent->get<fp16_t>();
            IE_ASSERT(scalesContentPtr != nullptr);

            for (int i = 0; i < scales->desc().totalDimSize(); ++i) {
                if (ie::PrecisionUtils::f16tof32(scalesContentPtr[i]) < SCALE_THRESHOLD) {
                    return false;
                }
            }
        }
    }

    return true;
}

int correctShift(int shift, bool firstStage, const std::string& type) {
    auto caselessEq = InferenceEngine::details::CaselessEq<std::string>();

    if (firstStage && shift > 10) {
        shift -= 8;
    }

    if (caselessEq(type, "Convolution") || caselessEq(type, "Deconvolution")) {
        shift = std::min(shift, 8);
    } else if (caselessEq(type, "FullyConnected")) {
        shift = std::min(shift, 9);
    }

    return shift;
}

int maxOutputExponent(const std::string& name, const InferenceEngine::NetworkStatsMap& stats) {
    auto node_stats_it = stats.find(name);
    IE_ASSERT(node_stats_it != stats.end());

    auto& max = node_stats_it->second->_maxOutputs;
    auto& min = node_stats_it->second->_maxOutputs;

    IE_ASSERT(max.size() > 0 && min.size() > 0);
    auto max_value = *std::max_element(max.begin(), max.end());
    auto min_value = *std::min_element(min.begin(), min.end());

    max_value = std::max(fabsf(max_value), fabsf(min_value));
    IE_ASSERT(max_value > 0);
    int exp = 0;

    // frexp fractions float into two parts:
    // [0.5, 1)* 2^exp
    // while float stores value in format
    // [1, 2) * 2^f_exp
    // which means exp returned by frexp is f_exp + 1
    frexp(max_value, &exp);
    return exp - 1;
}

void scaleBlobByIdx(const Model& model, const Stage& stage, int index, float scale) {
    const auto& original = stage->input(index);
    IE_ASSERT(original->usage() == DataUsage::Fake || original->usage() == DataUsage::Const);
    if (original->usage() != DataUsage::Const) {
        return;
    }

    auto scaled = model->duplicateData(original, "@scaled", original->desc(), scaleContent(original->content(), scale));
    scaled->attrs().set<float>("scaleFactor", scale);
    model->replaceStageInput(stage->inputEdge(index), scaled);
}

void addScaleInput(const Model& model, const Stage& stage, float scale) {
    static constexpr int SCALES_IDX = 3;

    IE_ASSERT(stage->numOutputs() == 1);
    IE_ASSERT(stage->output(0)->desc().dims().has(Dim::C));
    const auto outputChannels = stage->output(0)->desc().dims()[Dim::C];

    auto scaleInput = model->addConstData(stage->name() + "@scales", DataDesc{{outputChannels}}, replicateContent(1.0f / scale, outputChannels));
    model->replaceStageInput(stage->inputEdge(SCALES_IDX), scaleInput);
}

void scaleWeightableStage(const Model& model, const Stage& stage, float factor) {
    IE_ASSERT(stage->numInputs() == 4);

    if (factor == 1) {
        return;
    }

    static constexpr int WEIGHTS_IDX = 1;
    static constexpr int BIASES_IDX  = 2;

    scaleBlobByIdx(model, stage, WEIGHTS_IDX, factor);
    scaleBlobByIdx(model, stage, BIASES_IDX, factor);

    addScaleInput(model, stage, factor);

    stage->attrs().set<float>("scaleFactor", factor);
}

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(analyzeWeightableLayers);

    static const int scaleToExp     = 8;  // get from config?
    static const int scaleThreshold = 1;

    auto& stats  = model->nodesStats();

    bool isGrowingOutput = checkGrowingOutput(model);

    bool firstStage = true;
    int  normalVal  = 0;
    const auto& env = CompileEnv::get();

    for (const auto& stage : model->getStages()) {
        if (!isScalable(stage)) {
            continue;
        }
        IE_ASSERT(stage->origLayer() != nullptr);

        // Get scale from IR, compute if it was absent
        auto scale = stage->origLayer()->GetParamAsFloat("vpu_scale", 0);
        if (!scale) {
            auto weights = stage->input(1);

            auto content = weights->content();
            IE_ASSERT(content != nullptr);

            auto weightsVals = content->get<fp16_t>();
            IE_ASSERT(weightsVals != nullptr);

            auto exponents = calculateExponents(weightsVals, weights->desc().totalDimSize());

            int maxExp = *std::max_element(exponents.begin(), exponents.end());
            int shift = largestExp - maxExp;

            auto meanExp = getMeanValue(exponents);
            shift = std::min(-meanExp, shift);

            if (stats.empty()) {
                if (firstStage && shift < 4 && isGrowingOutput && weights->desc().dim(Dim::C) > 1) {
                    normalVal = 5;
                }

                shift = correctShift(shift, firstStage, stage->origLayer()->type);
                shift -= normalVal;
            } else {
                int outExp = maxOutputExponent(stage->origLayer()->name, stats);  // what if outExp == 15?
                shift = std::min(scaleToExp - outExp, shift);
            }

            firstStage = false;
            scale = 1;
            if (shift > scaleThreshold) {
                scale = static_cast<float>(1ULL << static_cast<std::uint32_t>(shift));
            }

            if (!env.config.irWithVpuScalesDir.empty()) {
                stage->origLayer()->params["vpu_scale"] = toString(scale);
            }
        }
        scaleWeightableStage(model, stage, scale);
    }
}

}  // namespace

Pass::Ptr PassManager::analyzeWeightableLayers() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
