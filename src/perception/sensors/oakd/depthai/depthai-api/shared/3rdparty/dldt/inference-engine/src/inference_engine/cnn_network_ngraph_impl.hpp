// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <ie_icnn_network.hpp>
#include <map>
#include <memory>
#include <ngraph/attribute_visitor.hpp>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <string>
#include <vector>

#include "cnn_network_impl.hpp"
#include "description_buffer.hpp"
#include "ie_api.h"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"
#include "ie_input_info.hpp"

namespace InferenceEngine {
namespace ShapeInfer {
class Reshaper;

using ReshaperPtr = std::shared_ptr<Reshaper>;
}  // namespace ShapeInfer
namespace details {

class INFERENCE_ENGINE_API_CLASS(CNNNetworkNGraphImpl): public ICNNNetwork {
public:
    CNNNetworkNGraphImpl(const std::shared_ptr<::ngraph::Function>& nGraph);
    ~CNNNetworkNGraphImpl() override;

    Precision getPrecision() const noexcept override;
    void getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept override;

    void getInputsInfo(InputsDataMap& inputs) const noexcept override;

    InputInfo::Ptr getInput(const std::string& inputName) const noexcept override;

    void getName(char* pName, size_t len) const noexcept override;

    const std::string& getName() const noexcept override;

    size_t layerCount() const noexcept override;

    void setInputInfo(InputInfo::Ptr data);

    DataPtr& getData(const char* name) noexcept override;

    DataPtr& getData(const std::string& name) {
        return getData(name.c_str());
    }

    std::shared_ptr<ICNNNetwork> getCNNNetwork();

    // This method is not really implemented; don't call it
    void addLayer(const CNNLayerPtr& layer) noexcept override;

    StatusCode getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept override;

    // public version
    StatusCode setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept override;

    // for internal usage (e.g. setBatch via reshape in tests)
    StatusCode setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept;

    size_t getBatchSize() const noexcept override;

    StatusCode addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept override;

    void addOutput(const std::string& dataName);

    StatusCode getStats(ICNNNetworkStats** stats, ResponseDesc* resp) const noexcept override {
        if (cnnNetwork) {
            return cnnNetwork->getStats(stats, resp);
        }
        if (stats == nullptr) return StatusCode::PARAMETER_MISMATCH;
        *stats = _stats.get();
        return StatusCode::OK;
    }

    void Release() noexcept override {
        delete this;
    }

    const std::shared_ptr<const ::ngraph::Function> getFunction() const noexcept override {
        return networksEqual ? _ngraph_function : nullptr;
    }

    virtual void validate(int = 10);

    StatusCode reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                       ResponseDesc* resp) noexcept override;

    StatusCode AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                            InferenceEngine::ResponseDesc* resp) noexcept override;

    StatusCode serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
        noexcept override;

    void convertToCNNNetworkImpl();

    std::shared_ptr<CNNNetworkNGraphImpl> cloneNGraphImpl() const;
    void transformConstants();
protected:
    std::shared_ptr<::ngraph::Function> _ngraph_function;
    virtual std::shared_ptr<::ngraph::Function> cloneFunction(bool constFolding = false, const std::map<std::string,
            std::vector<size_t>>& inputShapes = {}) const;
private:
    std::map<std::string, DataPtr> _data;
    InferenceEngine::InputsDataMap _inputData;
    std::map<std::string, DataPtr> _outputData;
    CNNNetworkStatsImplPtr _stats;
    std::shared_ptr<CNNNetworkImpl> cnnNetwork;
    std::shared_ptr<::ngraph::Function> _converted_function;
    // If CNNNetwork and nGraph function have the same layers this flag is true
    // But this flag drops if CNNNetwork was changed (addLayer or something like this)
    bool networksEqual = true;

    /**
     * @brief Create DataPtr for nGraph operation
     *
     * @param output output port from nGraph op
     * @param outName name for DataPtr
     * @param ptr reference to new DataPtr
     */
    void createDataForResult(const ::ngraph::Output<::ngraph::Node>& output, const std::string& outName, DataPtr& ptr);
    void convertFunctionToICNNNetwork(std::shared_ptr<::ngraph::Function>& graph, std::shared_ptr<CNNNetworkImpl>& cnnNetwork) const;

    /**
     * @brief Reshape on the same shape
     */
    void reshape();

    bool has_f16_constants(const std::shared_ptr<::ngraph::Function> &function) const;
};

class TINGraphBody : public CNNNetworkNGraphImpl {
public:
    explicit TINGraphBody(const std::shared_ptr<::ngraph::Function>& func): CNNNetworkNGraphImpl(func) {}

protected:
    std::shared_ptr<::ngraph::Function> cloneFunction(bool constFolding, const std::map<std::string, std::vector<size_t>>& inputShapes) const override {
        return _ngraph_function;
    }
};

class NGraphData : public Data {
public:
    using Ptr = std::shared_ptr<NGraphData>;

    NGraphData(CNNNetworkNGraphImpl* network, const std::string& name, const TensorDesc& desc)
        : Data(name, desc), network(network) {}

    void reset() {
        network = nullptr;
    }

    CNNLayerWeakPtr& getCreatorLayer() override {
        if (Data::getCreatorLayer().lock() == nullptr && network != nullptr) {
            network->convertToCNNNetworkImpl();
        }
        return Data::getCreatorLayer();
    }

    std::map<std::string, CNNLayerPtr>& getInputTo() override {
        if (Data::getInputTo().empty() && network != nullptr) {
            network->convertToCNNNetworkImpl();
        }

        return Data::getInputTo();
    }

private:
    CNNNetworkNGraphImpl* network;
};

/**
 * @brief Creator for CNNLayer from nGraph op
 */
class CNNLayerCreator : public ::ngraph::AttributeVisitor {
public:
    using CreatorFor = std::function<CNNLayerPtr(const std::shared_ptr<::ngraph::Node>& node,
                                                 const std::map<std::string, std::string> param)>;
    explicit CNNLayerCreator(const std::shared_ptr<::ngraph::Node>& node);

    CNNLayerPtr create();

    void on_attribute(const std::string& name, std::string& value) override {
        params[name] = value;
    }

    void on_attribute(const std::string& name, bool& value) override {
        params[name] = value ? "true" : "false";
    }

    void addSpecificCreator(const std::vector<std::string>& forTypes, const CreatorFor& creator) {
        for (const auto type : forTypes) {
            creators[type] = creator;
        }
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::string>& adapter) override {
        std::string data = adapter.get();
        std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        params[name] = data;
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        std::string dims;
        auto shape = adapter.get();
        for (size_t i = 0; i < shape.size(); i++) {
            if (!dims.empty()) dims += ",";
            dims += std::to_string(shape[i]);
        }
        params[name] = dims;
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<double>& adapter) override {
        params[name] = std::to_string(adapter.get());
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<int64_t>& adapter) override {
        params[name] = std::to_string(adapter.get());
    }

    void on_adapter(const std::string& name, ::ngraph::ValueAccessor<void>& adapter) override;

private:
    std::shared_ptr<::ngraph::Node> node;
    std::map<std::string, std::string> params;
    std::map<std::string, CreatorFor> creators;
};

typedef std::shared_ptr<CNNNetworkNGraphImpl> CNNNetworkNGraphImplPtr;
}  // namespace details
}  // namespace InferenceEngine
