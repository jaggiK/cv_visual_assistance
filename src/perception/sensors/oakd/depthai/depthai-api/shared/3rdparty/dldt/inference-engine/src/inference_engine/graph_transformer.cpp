// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer.h"

#include <cpp/ie_cnn_network.h>
#include <details/ie_cnn_network_tools.h>

#include <details/caseless.hpp>
#include <iterator>
#include <map>
#include <utility>
#include <memory>
#include <shape_infer/const_infer/ie_const_infer_holder.hpp>
#include <string>
#include <vector>
#include <mutex>

#include "blob_factory.hpp"
#include "cnn_network_impl.hpp"
#include "graph_tools.hpp"
#include "net_pass.h"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace InferenceEngine {

bool isForFakeQuantzie(const CNNLayer& layer) {
    for (const DataPtr data : layer.outData) {
        for (const auto it : data->getInputTo()) {
            const CNNLayerPtr childLayer = it.second;
            if (childLayer->type == "FakeQuantize" || childLayer->type == "Quantize") {
                return true;
            }
        }
    }

    return false;
}

static std::vector<DataPtr> get_inputs(details::CNNNetworkImpl* _network) {
    if (!_network) return {};

    InputsDataMap ins_info;
    _network->getInputsInfo(ins_info);

    std::vector<DataPtr> inputs;
    for (const auto& kvp : ins_info)
        inputs.push_back(kvp.second->getInputData());
    return inputs;
}

static std::vector<DataPtr> get_outputs(details::CNNNetworkImpl* _network) {
    if (!_network) return {};

    std::map<std::string, DataPtr> outs_info;
    _network->getOutputsInfo(outs_info);

    std::vector<DataPtr> outputs;
    for (const auto& kvp : outs_info)
        outputs.push_back(kvp.second);
    return outputs;
}

ConstTransformer::ConstTransformer(details::CNNNetworkImpl* _network)
        : inputs(get_inputs(_network)), outputs(get_outputs(_network)), network(_network) {
    if (!_network)
        THROW_IE_EXCEPTION << "[ERROR]: Failed to init ConstTransformer with null pointer of network";
}

ConstTransformer::ConstTransformer(std::vector<DataPtr> &_inputs, std::vector<DataPtr> &_outputs)
        : inputs(_inputs), outputs(_outputs), network(nullptr) {
    if (inputs.empty() || outputs.empty())
        THROW_IE_EXCEPTION << "[ERROR]: Failed to init ConstTransformer with empty list of inputs or outputs";
}

std::vector<CNNLayerPtr> ConstTransformer::foldConstSubgraphsInternal(const std::map<std::string, bool>& constLayers,
                                                                      const BlobMap& constData,
                                                                      const std::vector<CNNLayerPtr>& sortedLayers) {
    std::vector<CNNLayerPtr> remainingConstLayers;
    for (const auto& layer : sortedLayers) {
        if (constLayers.find(layer->name) != constLayers.end()) {
            // const layer doesn't need parent connections -> erase them
            for (const auto& insData : layer->insData) {
                auto& inputTo = insData.lock()->getInputTo();
                inputTo.erase(layer->name);
                // Note: to resolve corner case above layers can be marked as const with const data, just to be removed
                // properly.. and maybe this logic wouldn't be needed
                if (inputTo.empty()) {
                    auto creator = insData.lock()->getCreatorLayer().lock();
                    auto it = std::find(creator->outData.begin(), creator->outData.end(), insData.lock());
                    if (it != creator->outData.end()) {
                        data_to_remove.push_back(*it);
                        creator->outData.erase(it);
                    }
                }
            }
            layer->insData.clear();

            if (constLayers.at(layer->name)) {
                for (const auto& outData : layer->outData) {
                    for (const auto& inputTo : outData->getInputTo()) {
                        CNNLayerPtr inputToLayer;
                        std::string inputToName;
                        std::tie(inputToName, inputToLayer) = inputTo;
                        auto& insData = inputToLayer->insData;
                        auto insDataIt =
                            std::find_if(insData.begin(), insData.end(), [&outData](const DataWeakPtr& current) {
                                return current.lock()->getName() == outData->getName();
                            });
                        // remove connection with const data, because for const child it's not needed, for dynamic - new
                        // one will be created
                        if (insDataIt != insData.end()) {
                            insDataIt = inputToLayer->insData.erase(insDataIt);
                        }
                    }
                    data_to_remove.push_back(outData);
                }
                layer_to_remove.push_back(layer);
            } else {
                // if only one output data is not const - do nothing, otherwise - run procedure below
                // note: multiple const output data requires multiple layers with blob["custom"] to keep const data
                bool keepConstData = layer->outData.size() == 1;
                if (keepConstData) {
                    auto outData = layer->outData[0];
                    for (const auto& inputTo : outData->getInputTo()) {
                        if (constLayers.find(inputTo.first) != constLayers.end()) {
                            keepConstData = false;
                        }
                    }
                }
                if (keepConstData) {
                    if (!constLayers.at(layer->name)) {
                        auto outData = layer->outData[0];
                        if (layer->blobs.find("custom") == layer->blobs.end()) {
                            // if there's no const data - set it
                            const auto it = constData.find(outData->getName());
                            if (it != constData.end()) {
                                layer->blobs["custom"] = it->second;
                            }
                        }
                        if (layer->type != "Const") {
                            // layer was calculated during the Const Propagation, need to hide its semantic (type,
                            // params)
                            LayerParams layerParams {layer->name + "__" + outData->getName() + "__Const", "Const",
                                                     layer->precision};
                            auto newLayer = std::make_shared<CNNLayer>(layerParams);
                            for (const auto& data : layer->outData) {
                                data->getCreatorLayer() = newLayer;
                            }
                            newLayer->outData = layer->outData;
                            newLayer->blobs["custom"] = layer->blobs["custom"];
                            layer_to_remove.push_back(layer);
                            layer_to_add.push_back(newLayer);
                            remainingConstLayers.push_back(newLayer);
                        } else {
                            // Layer with `Const` type should be also considered on trimming shape inputs
                            remainingConstLayers.push_back(layer);
                        }
                    }
                } else {
                    for (const auto& outData : layer->outData) {
                        for (const auto& inputTo : outData->getInputTo()) {
                            CNNLayerPtr inputToLayer;
                            std::string inputToName;
                            std::tie(inputToName, inputToLayer) = inputTo;
                            auto& insData = inputToLayer->insData;
                            auto insDataIt =
                                std::find_if(insData.begin(), insData.end(), [&outData](const DataWeakPtr& current) {
                                    return current.lock()->getName() == outData->getName();
                                });
                            // remove connection with const data, because for const child it's not needed, for dynamic -
                            // new one will be created
                            if (insDataIt != insData.end()) {
                                insDataIt = inputToLayer->insData.erase(insDataIt);
                            }
                            if (constLayers.find(inputToName) == constLayers.end()) {
                                // next layer is not const, need to attach const data to it via blobs["custom"] of new
                                // Const layer
                                LayerParams layerParams {layer->name + "__" + outData->getName() + "__Const", "Const",
                                                         layer->precision};
                                auto newLayer = std::make_shared<CNNLayer>(layerParams);
                                remainingConstLayers.push_back(newLayer);
                                const auto it = constData.find(outData->getName());
                                if (it != constData.end()) {
                                    newLayer->blobs["custom"] = it->second;
                                }
                                auto newData = std::make_shared<Data>(outData->getName() + "__" + inputToName,
                                                                      outData->getTensorDesc());
                                newData->getCreatorLayer() = newLayer;
                                newData->getInputTo()[inputToName] = inputToLayer;
                                newLayer->outData = {newData};
                                layer_to_add.push_back(newLayer);
                                data_to_add.push_back(newData);
                                inputToLayer->insData.insert(insDataIt, newData);
                            }
                        }
                    }
                    for (const auto& data : layer->outData) {
                        data_to_remove.push_back(data);
                    }
                    layer_to_remove.push_back(layer);
                }
            }
        }
        if (NetPass::HasInternalSubnet(layer)) {
            auto subgraph = NetPass::GetInternalSubnet(layer);
            ConstTransformer transformer(subgraph.inputs, subgraph.outputs);
            transformer.foldConstSubgraphs();
        }
    }
    return remainingConstLayers;
}

const std::map<std::string, bool> ConstTransformer::getConstLayers(const std::vector<CNNLayerPtr>& sortedLayers) {
    std::map<std::string, bool> mapConstLayers;
    // collect all const layers, which inputs are const layers.
    for (const auto& layer : sortedLayers) {
        // Layers with "Shape" and "Const" type are Const by definition
        if (layer->type == "Shape" || layer->type == "Const") {
            mapConstLayers[layer->name] = false;
        } else if ((layer->type != "FakeQuantize") && (layer->type != "Quantize") && (!isForFakeQuantzie(*layer))) {
            bool isAllInputsConst = true;
            for (auto const& data : layer->insData) {
                auto creator = data.lock()->getCreatorLayer().lock();
                if (creator != nullptr) {
                    if (mapConstLayers.find(creator->name) == mapConstLayers.end()) {
                        isAllInputsConst = false;
                    }
                } else {
                    // Empty creator means that it's a network representation via inputs/outs data collection
                    // And it's a firs layer in network.
                    isAllInputsConst = false;
                }
            }
            if (isAllInputsConst && !layer->insData.empty()) mapConstLayers[layer->name] = false;
        }
    }
    // Add mark for const layers, if it's used for shape taking layers as second input
    // true - is used and can be deleted from graph, as no influence on data, false - opposite
    std::map<std::string, bool> mapVisitedLayers = mapConstLayers;
    for (auto rit = sortedLayers.rbegin(); rit != sortedLayers.rend(); rit++) {
        auto currentLayer = (*rit);
        std::string currentLayerName = currentLayer->name;
        bool isCurrentConst = mapConstLayers.find(currentLayerName) != mapConstLayers.end();
        for (int i = 0; i < currentLayer->insData.size(); i++) {
            std::string creatorName;
            if (currentLayer->insData[i].lock() != nullptr) {
                auto creator = currentLayer->insData[i].lock()->getCreatorLayer().lock();
                if (creator) {
                    creatorName = creator->name;
                }
            }
            bool isCreatorConst = mapConstLayers.find(creatorName) != mapConstLayers.end();
            if (isCreatorConst) {
                // mark second const input of shape taking layers (Reshape, Interp..), if they wasn't visited before
                if ((i == 1) && (shapeTaking.find(currentLayer->type)) != shapeTaking.end()) {
                    if (!mapConstLayers[creatorName]) {
                        if (!mapVisitedLayers.at(creatorName)) {
                            mapConstLayers[creatorName] = true;
                        }
                    }
                } else {
                    if (isCurrentConst) {
                        if (mapConstLayers.at(currentLayerName)) {
                            if (!mapConstLayers[creatorName]) {
                                if (!mapVisitedLayers.at(creatorName)) {
                                    mapConstLayers[creatorName] = true;
                                }
                            }
                        } else {
                            mapConstLayers[creatorName] = false;
                        }
                    } else {
                        mapConstLayers[creatorName] = false;
                    }
                }
            }
            mapVisitedLayers[creatorName] = true;
        }
        mapVisitedLayers[currentLayerName] = true;
    }
    return mapConstLayers;
}

const BlobMap ConstTransformer::getConstData(const std::map<std::string, bool>& constLayers,
                                             const std::vector<CNNLayerPtr>& sortedLayers) {
    ShapeInfer::ConstInferHolder holder;
    BlobMap constData;
    auto getInputBlobs = [&constData](const std::vector<DataWeakPtr>& insData,
                                      bool isForShape) -> std::vector<Blob::CPtr> {
        std::vector<Blob::CPtr> inputBlobs;
        // special case of Const layers: no inputs, no input blobs
        if (insData.empty()) {
            return {};
        }
        for (const auto& data : insData) {
            std::string dataName = data.lock()->getName();
            if (constData.find(dataName) != constData.end()) {
                // get blobs, inferred before
                inputBlobs.push_back(constData.at(dataName));
            } else {
                // special case of Shape layer: no input data, but blob contains info about dimensions, layout and
                // etc...
                auto blob = make_blob_with_precision(data.lock()->getTensorDesc());
                inputBlobs.push_back(blob);
            }
        }
        return inputBlobs;
    };

    auto getOutputBlobs = [](const std::vector<DataPtr>& outData) -> std::vector<Blob::Ptr> {
        std::vector<Blob::Ptr> outputBlobs;
        for (const auto& data : outData) {
            auto blob = make_blob_with_precision(data->getTensorDesc());
            blob->allocate();
            outputBlobs.push_back(blob);
        }
        return outputBlobs;
    };

    for (const auto& layer : sortedLayers) {
        if (layer->type == "FakeQuantize" || layer->type == "Quantize") {
            continue;
        }

        if (constLayers.find(layer->name) != constLayers.end()) {
            std::string layerName = layer->name;
            bool isForShape = constLayers.at(layerName);

            auto implPtr = holder.getConstInferImpl(layer->type);
            if (!implPtr && !isForShape)
                if (layer->type != "FakeQuantize" && layer->type != "Quantize")
                    THROW_IE_EXCEPTION << "Failed to find reference implementation for `" + layer->name +
                                              "` Layer with `" + layer->type + "` Type on constant propagation";
            if (!isForShape) {
                auto outputBlobs = getOutputBlobs(layer->outData);
                auto inp = getInputBlobs(layer->insData, isForShape);
                if (layer->type != "FakeQuantize" && layer->type != "Quantize")
                    implPtr->infer(inp, layer->params, layer->blobs, outputBlobs);
                for (int i = 0; i < layer->outData.size(); i++) {
                    std::string dataName = layer->outData[i]->getName();
                    auto shapes = layer->outData[i]->getTensorDesc().getDims();
                    outputBlobs[i]->getTensorDesc().reshape(shapes, TensorDesc::getLayoutByDims(shapes));
                    constData[dataName] = outputBlobs[i];
                }
            }
        }
    }
    return constData;
}

/**
 * Will replace provided layer with reshape with corresponding shape from output data
 *
 * @param layer is operation to replace with static reshape
 * @return newly created reshape static layer
 */
static CNNLayerPtr replace_with_static_reshape(CNNLayerPtr &layer) {
    IE_ASSERT(layer->insData.size() == 1);
    IE_ASSERT(layer->outData.size() == 1);

    auto in_data = layer->insData[0].lock();
    if (in_data == nullptr)
        THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has invalid input data";
    auto out_data = layer->outData[0];

    auto precision = out_data->getPrecision();
    auto shape = out_data->getDims();

    // TODO: Have to use old name instead a new one because tensor statistic is mapped
    //       to layers by name. The old int8 pipeline may be broken because of lose
    //       tensor statistic for particular reshape.
    auto reshape = std::make_shared<ReshapeLayer>(
            LayerParams{layer->name, "Reshape", precision});
    reshape->shape = std::vector<int>(shape.begin(), shape.end());

    // replacement
    auto &input_to_map = in_data->getInputTo();

    // try to find by name
    auto found_by_name = input_to_map.find(layer->name);
    if (found_by_name != input_to_map.end()) {
        input_to_map.erase(found_by_name);
    } else {
        // try to find by ptr
        auto found_by_ptr = std::find_if(input_to_map.begin(), input_to_map.end(),
                                         [&layer] (const std::pair<std::string, CNNLayerPtr> &p)
                                         { return p.second == layer; });
        if (found_by_ptr != input_to_map.end())
            input_to_map.erase(found_by_ptr);
    }
    input_to_map[reshape->name] = reshape;

    reshape->insData = {in_data};
    reshape->outData = {out_data};
    out_data->getCreatorLayer() = reshape;

    return reshape;
}

void ConstTransformer::trimShapeInputs(const std::vector<CNNLayerPtr>& constLayers,
                                       std::vector<CNNLayerPtr>& allLayers) {
    for (const auto& layer : constLayers) {
        if (layer->outData.size() == 1 && layer->type == "Const" && layer->insData.empty()) {
            auto constData = layer->outData[0];
            std::map<std::string, CNNLayerPtr> inputToMap = constData->getInputTo();
            for (const auto& inputTo : inputToMap) {
                CNNLayerPtr inputToLayer = inputTo.second;
                if (shapeTaking.find(inputToLayer->type) != shapeTaking.end()) {
                    auto& insData = inputToLayer->insData;
                    auto it = std::find_if(insData.begin(), insData.end(), [&constData](const DataWeakPtr& current) {
                        return current.lock()->getName() == constData->getName();
                    });
                    if (it != insData.end() && std::distance(insData.begin(), it) == 1) {
                        inputToLayer->insData.erase(it);
                        constData->getInputTo().erase(inputTo.first);
                    }
                }
            }
            if (constData->getInputTo().empty()) {
                layer_to_remove.push_back(layer);
                data_to_remove.push_back(constData);
            }
        }
    }
    // TODO: Some WA. Previous step foldConstSubgraphsInternal remove all const data
    //       from graph. Although that is responsibility of trimShapeInputs pass.
    //       That's why we need make additional pass through allLayers and replace
    //       all shape taken layers like Squeeze/Flatten with Reshape with single input.
    for (auto& layer : allLayers) {
        // Layer is from list of reshape-like layers
        if (layer->type != "Reshape" &&
            layer->type != "Unsqueeze" &&
            layer->type != "Squeeze" &&
            layer->type != "Flatten")
            continue;

        // already removed
        if (std::find(layer_to_remove.begin(), layer_to_remove.end(), layer) != layer_to_remove.end())
            continue;

        // The second input was not removed. So shape is not constant.
        if (layer->insData.size() != 1)
            continue;

        auto new_one = replace_with_static_reshape(layer);
        layer_to_remove.push_back(layer);
        layer_to_add.push_back(new_one);
    }
}

void ConstTransformer::cleanup() {
    if (network) {
        for (const auto &layer : layer_to_remove) network->removeLayer(layer->name);
        for (const auto &data : data_to_remove) network->removeData(data->getName());

        for (const auto &layer : layer_to_add) network->addLayer(layer);
        for (const auto &data : data_to_add) network->addData(data->getName().c_str(), data);
    } else {
        // Subgraph case
        auto &const_holder = inputs.back();
        if (const_holder->getPrecision() == Precision::UNSPECIFIED) {
            auto &holder_map = const_holder->getInputTo();
            // Remove from const holder data object
            for (const auto &layer : layer_to_remove) {
                auto self_found = std::find_if(holder_map.begin(), holder_map.end(),
                        [&layer] (const std::pair<std::string, CNNLayerPtr> kvp) {
                    return kvp.second == layer;
                });

                if (self_found != holder_map.end()) {
                    holder_map.erase(self_found);
                }
            }
            // Add to const holder
            for (const auto &layer : layer_to_add) {
                holder_map[layer->name] = layer;
            }
        }
    }
}

void ConstTransformer::foldConstSubgraphs() {
    auto sortedLayers = details::CNNSubnetSortTopologically({inputs, outputs});
    auto constLayers = getConstLayers(sortedLayers);
    auto constData = getConstData(constLayers, sortedLayers);
    foldConstSubgraphsInternal(constLayers, constData, sortedLayers);

    cleanup();
}

void ConstTransformer::fullTrim() {
    // Avoid data races on one network instance
    static std::mutex lockFullTrim;
    std::lock_guard<std::mutex> lock(lockFullTrim);
    auto sortedLayers = details::CNNSubnetSortTopologically({inputs, outputs});
    auto constMapLayers = getConstLayers(sortedLayers);
    auto constData = getConstData(constMapLayers, sortedLayers);
    auto constLayers = foldConstSubgraphsInternal(constMapLayers, constData, sortedLayers);
    trimShapeInputs(constLayers, sortedLayers);

    for (auto &layer : sortedLayers) {
        if (NetPass::HasInternalSubnet(layer)) {
            auto subgraph = NetPass::GetInternalSubnet(layer);

            ConstTransformer transformer(subgraph.inputs, subgraph.outputs);
            auto ti_sortedLayers = details::CNNSubnetSortTopologically({subgraph.inputs, subgraph.outputs});
            auto ti_constMapLayers = transformer.getConstLayers(ti_sortedLayers);
            auto ti_constData = transformer.getConstData(ti_constMapLayers, ti_sortedLayers);
            auto ti_constLayers = transformer.foldConstSubgraphsInternal(ti_constMapLayers, ti_constData, ti_sortedLayers);
            transformer.trimShapeInputs(ti_constLayers, ti_sortedLayers);
            transformer.cleanup();
        }
    }

    cleanup();
}
}  // namespace InferenceEngine
