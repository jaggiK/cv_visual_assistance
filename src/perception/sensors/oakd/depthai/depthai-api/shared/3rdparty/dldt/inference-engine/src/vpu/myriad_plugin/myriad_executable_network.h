// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <sstream>
#include <fstream>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/ie_executor_manager.hpp>

#include <vpu/graph_transformer.hpp>
#include <vpu/parsed_config.hpp>

#include "myriad_executor.h"
#include "myriad_infer_request.h"
#include "myriad_async_infer_request.h"
#include "myriad_config.h"

namespace vpu {
namespace MyriadPlugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<ExecutableNetwork> Ptr;

    explicit ExecutableNetwork(InferenceEngine::ICNNNetwork &network,
                               std::vector<DevicePtr> &devicePool,
                               const MyriadConfig& config);

    explicit ExecutableNetwork(std::istream& strm,
                               std::vector<DevicePtr> &devicePool,
                               const MyriadConfig& config);

    explicit ExecutableNetwork(const std::string &blobFilename,
                               std::vector<DevicePtr> &devicePool,
                               const MyriadConfig& config);


    virtual ~ExecutableNetwork() {
        try {
            _executor->deallocateGraph(_device, _graphDesc);
        }
        catch (...) {
            std::cerr << "ERROR ~ExecutableNetwork():\n"
                      << "Some errors occurred during the calling of the deallocateGraph() method";
        }
    }

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override {
        if (_device == nullptr || !_device->isBooted()) {
            THROW_IE_EXCEPTION << "Can not create infer request: there is no available devices with platform "
                               << _device->_platform;
        }

        return std::make_shared<MyriadInferRequest>(_graphDesc, networkInputs, networkOutputs,
                                                    _inputInfo, _outputInfo,
                                                    _graphMetaData.stagesMeta, _config, _log, _executor);
    }

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override {
        if (_device == nullptr || !_device->isBooted()) {
            THROW_IE_EXCEPTION << "Can not create infer request: there is no available devices with platform "
                               << _device->_platform;
        }

        auto syncRequestImpl = std::make_shared<MyriadInferRequest>(_graphDesc, _networkInputs, _networkOutputs,
                                                                    _inputInfo, _outputInfo,
                                                                    _graphMetaData.stagesMeta, _config, _log,
                                                                    _executor);
        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        auto taskExecutorGetResult = getNextTaskExecutor();
        auto asyncTreadSafeImpl = std::make_shared<MyriadAsyncInferRequest>(
                syncRequestImpl, _taskExecutor, _callbackExecutor, taskExecutorGetResult);
        asyncRequest.reset(new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(
                           asyncTreadSafeImpl),
                           [](InferenceEngine::IInferRequest *p) { p->Release(); });
        asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    }

    void Export(std::ostream& model) override {
        model.write(_graphBlob.data(), _graphBlob.size());
    }

    void Export(const std::string &modelFileName) override {
        std::ofstream modelFile(modelFileName, std::ios::out | std::ios::binary);

        if (modelFile.is_open()) {
            Export(modelFile);
        } else {
            THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export";
        }
    }

    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    void GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) override;

    void GetMappedTopology(
            std::map<std::string, std::vector<InferenceEngine::PrimitiveInfo::Ptr>> &deployedTopology) override {
        THROW_IE_EXCEPTION << "GetMappedTopology is not implemented\n";
    }

    void Import(std::istream& strm,
                std::vector<DevicePtr> &devicePool,
                const MyriadConfig& config);

private:
    Logger::Ptr _log;
    MyriadExecutorPtr _executor;
    std::vector<char> _graphBlob;
    GraphDesc _graphDesc;
    DevicePtr _device;
    GraphMetaInfo _graphMetaData;
    MyriadConfig _config;
    int _actualNumExecutors = 0;
    std::vector<std::string> _supportedMetrics;

    DataInfo _inputInfo;
    DataInfo _outputInfo;

    const size_t _maxTaskExecutorGetResultCount = 1;
    std::queue<std::string> _taskExecutorGetResultIds;

    ExecutableNetwork(std::vector<DevicePtr> &devicePool,
                      const MyriadConfig& config);

    InferenceEngine::ITaskExecutor::Ptr getNextTaskExecutor() {
        std::string id = _taskExecutorGetResultIds.front();

        _taskExecutorGetResultIds.pop();
        _taskExecutorGetResultIds.push(id);

        InferenceEngine::ExecutorManager *executorManager = InferenceEngine::ExecutorManager::getInstance();
        InferenceEngine::ITaskExecutor::Ptr taskExecutor = executorManager->getExecutor(id);

        return taskExecutor;
    }

    InferenceEngine::ICNNNetwork::Ptr buildRuntimeGraph(GraphMetaInfo& graphMetaInfo);
};

}  // namespace MyriadPlugin
}  // namespace vpu
