// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <unordered_map>
#include <set>
#include <utility>

#include <ie_icnn_network.hpp>
#include <details/caseless.hpp>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/optional.hpp>

namespace vpu {

namespace ie = InferenceEngine;

//
// CompilationConfig
//

VPU_DECLARE_ENUM(Platform,
    UNKNOWN = 0,
    MYRIAD_2 = 2450,
    MYRIAD_X = 2480
)

struct CompilationConfig final {
    //
    // Compilation options
    //

    int numSHAVEs = -1;
    int numCMXSlices = -1;

    bool hwOptimization = true;

    bool ignoreIRStatistic = false;

    std::string irWithVpuScalesDir;

    std::string customLayers;

    bool detectBatch = true;

    Optional<bool> copyOptimization;
    Optional<bool> injectSwOps;
    Optional<bool> packDataInCmx;
    bool mergeHwPoolToConv = true;
    bool hwDilation = false;
    bool forceDeprecatedCnnConversion = false;

    std::map<std::string, std::vector<int>> ioStrides;

    //
    // Debug options
    //

    ie::details::caseless_set<std::string> hwWhiteList;
    ie::details::caseless_set<std::string> hwBlackList;

    bool hwDisabled(const std::string& layerName) const {
        if (!hwWhiteList.empty()) {
            return hwWhiteList.count(layerName) == 0;
        }

        if (!hwBlackList.empty()) {
            return hwBlackList.count(layerName) != 0;
        }

        return false;
    }

    ie::details::caseless_set<std::string> noneLayers;

    bool skipAllLayers() const {
        if (noneLayers.size() == 1) {
            const auto& val = *noneLayers.begin();
            return val == "*";
        }
        return false;
    }

    bool skipLayerType(const std::string& layerType) const {
        return noneLayers.count(layerType) != 0;
    }
    bool ignoreUnknownLayers = false;

    std::string dumpInternalGraphFileName;
    std::string dumpInternalGraphDirectory;
    bool dumpAllPasses;

    bool disableReorder = false;  // TODO: rename to enableReorder and switch logic.
    bool enablePermuteMerging = true;
    bool enableReplWithSCRelu = false;
    bool enableReplaceWithReduceMean = true;

    //
    // Deprecated options
    //

    float inputScale = 1.0f;
    float inputBias = 0.0f;
};


//
// DataInfo
//

struct DataInfo final {
    std::unordered_map<std::string, int> offset;
    int totalSize = 0;
};

//
// CompiledGraph
//

struct CompiledGraph final {
    using Ptr = std::shared_ptr<CompiledGraph>;

    std::vector<char> blob;
    std::pair<char*, size_t> blobHeader;

    std::string networkName;

    int networkBatch = 0;

    GraphMetaInfo graphMeta;
    int numActiveStages = 0;

    DataInfo inputInfo;
    DataInfo outputInfo;

    int inputBufSize = 0;
    int outputBufSize = 0;

    std::uint32_t numShaves = 0;
    std::uint32_t numSlices = 0;
};

//
// compileNetwork
//

CompiledGraph::Ptr compileNetwork(
        ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log);

CompiledGraph::Ptr compileSubNetwork(
        ie::ICNNNetwork& network,
        const CompilationConfig& subConfig);

//
// getSupportedLayers
//

std::set<std::string> getSupportedLayers(
        const ie::ICNNNetwork& network,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log);

//
// Blob version and checks
//

const uint32_t BLOB_MAGIC_NUMBER  = 9709;
const uint32_t BLOB_VERSION_MAJOR = 5;
const uint32_t BLOB_VERSION_MINOR = 0;

}  // namespace vpu
