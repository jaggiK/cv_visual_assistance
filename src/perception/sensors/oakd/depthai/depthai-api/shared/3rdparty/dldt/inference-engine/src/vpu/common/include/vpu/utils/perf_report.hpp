// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <set>
#include <vector>
#include <memory>

#include <ie_common.h>
#include <ie_precision.hpp>
#include <ie_layouts.h>

#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;

struct StageMetaInfo final {
    ie::InferenceEngineProfileInfo::LayerStatus status = ie::InferenceEngineProfileInfo::LayerStatus::NOT_RUN;
    std::vector<ie::Precision> outPrecisions;
    std::vector<ie::Layout> outLayouts;

    int inputsNum = 0;

    std::string layerName;
    std::string layerType;

    std::string displayStageName;

    std::string stageName;
    std::string stageType;

    int execOrder = -1;
    float execTime = 0;
};

struct DataMetaInfo final {
    std::string name;
    ie::TensorDesc desc;
    size_t parentIndex;
    std::vector<size_t> childrenIndices;
};

struct GraphMetaInfo final {
    std::string graphName;
    std::vector<StageMetaInfo> stagesMeta;
    std::vector<DataMetaInfo> datasMeta;
};

VPU_DECLARE_ENUM(PerfReport,
    PerLayer,
    PerStage
)

std::map<std::string, ie::InferenceEngineProfileInfo> parsePerformanceReport(
        const std::vector<StageMetaInfo>& stagesMeta,
        const float* deviceTimings,
        int deviceTimingsCount,
        PerfReport perfReport,
        bool printReceiveTensorTime);

}  // namespace vpu
