// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "mkldnn_plugin.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_layers_dispatcher.hpp"
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <memory>
#include <ie_plugin_config.hpp>
#include <vector>
#include <tuple>

#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
#if defined(_WIN32) || defined(WIN32)
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>
#endif
#endif

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

NumaNodesWeights create_shared_weights_per_socket() {
    NumaNodesWeights _weightsSharing;
    std::vector<int> sockets = MKLDNNPlugin::cpu::getAvailableNUMANodes();
    for (auto s : sockets)
        _weightsSharing[s] = std::make_shared<MKLDNNWeightsSharing>();
    return _weightsSharing;
}

NumaNodesWeights Engine::weightsSharing = create_shared_weights_per_socket();
const SimpleDataHash MKLDNNWeightsSharing::simpleCRC;

Engine::Engine() {
    _pluginName = "CPU";
    addDefaultExtensions(extensionManager);
}

InferenceEngine::ExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(const ICore * /*core*/, InferenceEngine::ICNNNetwork &network, const std::map<std::string, std::string> &config) {
    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    for (const auto &ii : _networkInputs) {
        auto input_precision = ii.second->getPrecision();
        if (input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::U8) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    // TODO: handle input precision differently - per input and not one per network...

    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;
    conf.readProperties(config);

    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(network.getBatchSize());
    }

    return std::make_shared<MKLDNNExecNetwork>(network, conf, extensionManager);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    // accumulate config parameters on engine level
    engConfig.readProperties(config);
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    Parameter result;
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key " << name;
    }
    return result;
}

static bool hasAVX512() {
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
    unsigned int regs[4] = {7, 0, 0, 0};
#if defined(_WIN32) || defined(WIN32)
    __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#else
    __cpuid_count(regs[0], regs[1], regs[0], regs[1], regs[2], regs[3]);
#endif
    if (regs[1] & (1U << 16))
        return true;
#endif
    return false;
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string brand_string;
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
        unsigned int addr_list[3] = { 0x80000002, 0x80000003, 0x80000004 };
        unsigned int regs[4];
        for (auto addr : addr_list) {
            regs[0] = addr;
#if defined(_WIN32) || defined(WIN32)
            __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#else
            __get_cpuid(regs[0], &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
            char *ch = reinterpret_cast<char*>(&regs[0]);
            for (size_t j = 0; j < sizeof(regs); j++)
                brand_string += ch[j];
        }
#else
        brand_string = "Non Intel Architecture";
#endif
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, brand_string);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;
        if (hasAVX512())
            capabilities.push_back(METRIC_VALUE(WINOGRAD));
        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(INT8));
        capabilities.push_back(METRIC_VALUE(BIN));
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && opt : engConfig._config)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    extensionManager->AddExtension(extension);
}

void Engine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config, QueryNetworkResult& res) const {
    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));
    while (i != details::CNNNetworkIterator()) {
        try {
            mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));
            // if we can create and have not thrown exception, then layer is supported
            std::unique_ptr <MKLDNNNode>(MKLDNNNode::CreateNode(*i, eng, extensionManager));
            res.supportedLayersMap.insert({ (*i)->name, GetName() });
        } catch (InferenceEngine::details::InferenceEngineException&) {
        }
        i++;
    }
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(
                {{2, 1},
                 CI_BUILD_NUMBER,
                 "MKLDNNPlugin"}, std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

IE_SUPPRESS_DEPRECATED_END
