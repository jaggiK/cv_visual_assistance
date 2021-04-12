// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>

#include <vpu/utils/logger.hpp>
#include <vpu/utils/ie_helpers.hpp>

#include "myriad_executor.h"
#include "myriad_config.h"

namespace vpu {
namespace MyriadPlugin {

class MyriadInferRequest : public InferenceEngine::InferRequestInternal {
    MyriadExecutorPtr _executor;
    Logger::Ptr _log;
    std::vector<StageMetaInfo> _stagesMetaData;
    MyriadConfig _config;

    const DataInfo _inputInfo;
    const DataInfo _outputInfo;

    GraphDesc _graphDesc;
    std::vector<uint8_t> resultBuffer;
    std::vector<uint8_t> inputBuffer;

public:
    typedef std::shared_ptr<MyriadInferRequest> Ptr;

    explicit MyriadInferRequest(GraphDesc &_graphDesc,
                                InferenceEngine::InputsDataMap networkInputs,
                                InferenceEngine::OutputsDataMap networkOutputs,
                                DataInfo& compilerInputsInfo,
                                DataInfo& compilerOutputsInfo,
                                const std::vector<StageMetaInfo> &blobMetaData,
                                const MyriadConfig &myriadConfig,
                                const Logger::Ptr &log,
                                const MyriadExecutorPtr &executor);

    void InferImpl() override;
    void InferAsync();
    void GetResult();

    void
    GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;
};

}  // namespace MyriadPlugin
}  // namespace vpu
