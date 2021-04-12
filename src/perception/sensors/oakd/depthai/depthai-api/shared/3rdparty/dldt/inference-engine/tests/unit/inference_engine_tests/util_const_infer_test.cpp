// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <initializer_list>
#include <string>
#include <utility>
#include <unordered_set>
#include <unordered_map>

#include <ie_util_internal.hpp>
#include <tests_common.hpp>
#include <graph_transformer.h>
#include "blob_factory.hpp"
#include "debug.h"
#include "util_test.hpp"
#include "util_const_infer_test.hpp"
#include <details/ie_cnn_network_tools.h>
#include <precision_utils.h>

namespace IE = InferenceEngine;

void RemoveLayerTests::SetUp() {
    net = getNetwork();
    originalLayersNum = net->allLayers().size();
    testTransformator.reset(new ConstTransformatorTest(net.get()));
}

//
// I1-d1-L1-d4              I4
//       / \  \              \
//      |  d7  \            d10
//      |  |    \            /
//  I2-d2-L2-d5-L4-d6-L5-d9-L10
//        /           /
//       /  ____d8___/
//      /  /
// I3-d3-L3
//
IE::details::CNNNetworkImplPtr RemoveLayerTests::getNetwork() {
    return netBuilder
            .data("data1", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data2", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data3", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data4", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data5", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data6", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data7", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data8", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data9", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data10", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .data("data11", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 3 }, IE::Layout::CHW))
            .layer<IE::CNNLayer>(IE::LayerParams{"input1", "input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"input3", "input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"input4", "input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer1", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer2", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer3", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer4", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer5", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer6", "dummy", IE::Precision::FP32})
            .linkToData("input1", "data1")
            .linkToData("input2", "data2")
            .linkToData("input3", "data3")
            .linkToData("input4", "data10")

            .linkDataTo("data1", "layer1")
            .linkDataTo("data2", "layer2")
            .linkDataTo("data2", "layer1")
            .linkDataTo("data3", "layer3")
            .linkDataTo("data3", "layer2")
            .linkDataTo("data10", "layer6")

            .linkToData("layer1", "data4")
            .linkToData("layer1", "data7")
            .linkToData("layer2", "data5")
            .linkToData("layer3", "data8")

            .linkDataTo("data4", "layer4")
            .linkDataTo("data5", "layer4")
            .linkDataTo("data8", "layer5")
            .linkDataTo("data7", "layer2")

            .linkToData("layer4", "data6")

            .linkDataTo("data6", "layer5")

            .linkToData("layer5", "data9")

            .linkDataTo("data9", "layer6")

            .linkToData("layer6", "data11")

            .addInput("data1")
            .addInput("data2")
            .addInput("data3")
            .finalize();
}

IE::CNNLayerPtr RemoveLayerTests::getLayer(const std::string& name) {
    const auto& layers = netBuilder.getLayersMap();
    auto it = layers.find(name);
    if (it == layers.end()) throw std::logic_error("Failed to find layer: " + name);
    return it->second;
}

IE::DataPtr RemoveLayerTests::getData(const std::string& name) {
    const auto& datas = netBuilder.getDataMap();
    auto it = datas.find(name);
    if (it == datas.end()) throw std::logic_error("Failed to find data: " + name);
    return it->second;
}

IE::BlobMap RemoveLayerTests::fillConstData(const std::vector<std::string>& constLayers) {
    IE::BlobMap constData;
    for (const auto& name:constLayers) {
        auto layer = getLayer(name);
        for (const auto& outData:layer->outData) {
            IE::TensorDesc desc = outData->getTensorDesc();
            IE::Blob::Ptr blob = make_blob_with_precision(desc);
            blob->allocate();
            auto* buffer = blob->buffer().as<float*>();
            size_t buffer_length = blob->byteSize() / sizeof(float);
            for (int i = 0; i < buffer_length; i++) {
                buffer[i] = i + 1;
            }
            constData[outData->getName()] = blob;
        }
    }
    return constData;
}

IE::BlobMap RemoveLayerTests::initConstLayers(const std::vector<std::string>& constLayers) {
    for (const auto& name : constLayers) {
        getLayer(name)->type = "Const";
    }
    IE::BlobMap customBlobs = fillConstData(constLayers);
    for (const auto& layerName: constLayers) {
        auto layer = getLayer(layerName);
        layer->type = "Const";
        layer->blobs["custom"] = customBlobs[layer->outData[0]->getName()];
    }
    return customBlobs;
}

IE::BlobMap RemoveLayerTests::fillConstDataDiffPrec (const std::vector<std::string>& constLayers) {
    IE::BlobMap constData;
    for (const auto& name:constLayers) {
        auto layer = getLayer(name);
        for (const auto& outData:layer->outData) {
            IE::TensorDesc desc = outData->getTensorDesc();
            IE::Blob::Ptr blob = make_blob_with_precision(desc);
            blob->allocate();
            switch(layer->precision) {
                case IE::Precision::U8: {
                    auto *buffer = blob->buffer().as<uint8_t *>();
                    for (int i = 0; i < blob->size(); i++) {
                        buffer[i] = i + 2;
                    }
                    break;
                }
                case IE::Precision::I32: {
                    auto *buffer = blob->buffer().as<int *>();
                    for (int i = 0; i < blob->size(); i++) {
                        buffer[i] = i + 2;
                    }
                    break;
                }
                case IE::Precision::I64: {
                    auto *buffer = blob->buffer().as<long long int *>();
                    for (int i = 0; i < blob->size(); i++) {
                        buffer[i] = i + 2;
                    }
                    break;
                }
                case IE::Precision::FP16: {
                    auto *buffer = blob->buffer().as<IE::ie_fp16 *>();
                    float j = 0;
                    for (int i = 0; i < blob->size(); i++) {
                        buffer[i] = j + (float)2;
                        buffer[i] = IE::PrecisionUtils::f32tof16(buffer[i]);
                        j++;
                    }
                    break;
                }
                case IE::Precision::FP32: {
                    auto *buffer = blob->buffer().as<float *>();
                    for (int i = 0; i < blob->size(); i++) {
                        buffer[i] = i + 2;
                    }
                    break;
                }
                default:
                    THROW_IE_EXCEPTION << "Not supported data type";
            }
            constData[outData->getName()] = blob;
        }
    }
    return constData;
}

IE::BlobMap RemoveLayerTests::initConstLayersDiffPrec(const std::vector<std::string> &constLayers) {
    for (const auto& name : constLayers) {
        getLayer(name)->type = "Const";
    }
    IE::BlobMap customBlobs = fillConstDataDiffPrec(constLayers);
    for (const auto& layerName: constLayers) {
        auto layer = getLayer(layerName);
        layer->type = "Const";
        layer->blobs["custom"] = customBlobs[layer->outData[0]->getName()];
    }
    return customBlobs;
}

TEST_F(RemoveLayerTests, canTrimL2) {
    auto layer1 = getLayer("layer1");
    auto layer4 = getLayer("layer4");
    auto data2 = getData("data2");
    auto data3 = getData("data3");
    auto data7 = getData("data7");
    auto data5 = getData("data5");
    std::vector<std::string> constLayers = {"layer2"};
    std::vector<std::string> refNewLayers = {constLayers[0] + "__data5__Const"};
    auto constData = fillConstData(constLayers);
    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);

    auto newLayers = testTransformator->foldConstSubgraphsInternal({{constLayers[0], false}}, constData, sortedLayers);

    std::vector<std::string> newLayer_names;
    for (const auto& layer : newLayers) newLayer_names.push_back(layer->name);

    ASSERT_EQ(newLayer_names, refNewLayers);
    IE::CNNNetwork cnnNetwork(net);
    ASSERT_THROW(cnnNetwork.getLayerByName("layer2"), IE::NotFound);
    auto newLayer = cnnNetwork.getLayerByName(refNewLayers[0].c_str());
    ASSERT_EQ(newLayer->type, "Const");
    ASSERT_EQ(constData["data5"], newLayer->blobs.at("custom"));
    ASSERT_EQ(nullptr, net->getData("data7"));
    net->removeData("data7");
    ASSERT_EQ(net->allLayers().size(), originalLayersNum);
    ASSERT_EQ(data2->getInputTo().size(), 1);
    ASSERT_EQ(data2->getInputTo().find("layer1")->second, layer1);
    ASSERT_EQ(data5->getCreatorLayer().lock(), newLayer);
    ASSERT_EQ(layer4->insData.size(), 2);
    ASSERT_EQ(layer4->insData[1].lock(), data5);
    ASSERT_EQ(layer1->insData.size(), 2);
    ASSERT_EQ(layer1->insData[0].lock(), getData("data1"));
    ASSERT_EQ(layer1->insData[1].lock(), data2);
    ASSERT_EQ(layer1->outData.size(), 1);
    ASSERT_EQ(layer1->outData[0], getData("data4"));
    ASSERT_EQ(newLayer->outData.size(), 1);
    ASSERT_EQ(newLayer->outData[0], data5);
    ASSERT_EQ(data3->getInputTo().size(), 1);
    ASSERT_EQ(data3->getInputTo().find("layer3")->second, getLayer("layer3"));
}

TEST_F(RemoveLayerTests, canTrimI1andL1) {
    auto layer4 = getLayer("layer4");
    auto layer2 = getLayer("layer2");
    auto data2 = getData("data2");
    std::vector<std::string> constLayers = {"input1", "layer1"};
    std::map<std::string, bool> mapConstLayers;
    for (const auto& it : constLayers) {
        mapConstLayers[it] = false;
    }
    std::vector<std::string> refNewLayers = {(constLayers[1] + "__data4__Const"), (constLayers[1] + "__data7__Const")};

    auto constData = fillConstData(constLayers);
    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto newLayers = testTransformator->foldConstSubgraphsInternal(mapConstLayers, constData, sortedLayers);

    std::vector<std::string> newLayer_names;
    for (auto layer : newLayers) newLayer_names.push_back(layer->name);

    ASSERT_EQ(newLayer_names, refNewLayers);
    IE::CNNNetwork cnnNetwork(net);
    ASSERT_THROW(cnnNetwork.getLayerByName("input1"), IE::NotFound);
    ASSERT_THROW(cnnNetwork.getLayerByName("layer1"), IE::NotFound);
    auto newLayerD4 = cnnNetwork.getLayerByName(refNewLayers[0].c_str());
    auto newLayerD7 = cnnNetwork.getLayerByName(refNewLayers[1].c_str());
    auto newData4 = net->getData("data4__layer4");
    auto newData7 = net->getData("data7__layer2");
    ASSERT_EQ(newLayerD4->type, "Const");
    ASSERT_EQ(newLayerD7->type, "Const");
    ASSERT_EQ(constData["data4"], newLayerD4->blobs.at("custom"));
    ASSERT_EQ(constData["data7"], newLayerD7->blobs.at("custom"));
    ASSERT_EQ(nullptr, net->getData("data1"));
    net->removeData("data1");
    ASSERT_EQ(net->allLayers().size(), originalLayersNum);
    ASSERT_EQ(data2->getInputTo().size(), 1);
    ASSERT_EQ(data2->getInputTo().find("layer2")->second, layer2);
    ASSERT_EQ(newData4->getCreatorLayer().lock(), newLayerD4);
    ASSERT_EQ(newData7->getCreatorLayer().lock(), newLayerD7);
    ASSERT_EQ(newLayerD4->outData.size(), 1);
    ASSERT_EQ(newLayerD7->outData.size(), 1);
    ASSERT_EQ(newLayerD4->outData[0], newData4);
    ASSERT_EQ(newLayerD7->outData[0], newData7);
    ASSERT_EQ(layer4->insData.size(), 2);
    ASSERT_EQ(layer4->insData[0].lock(), newData4);
    ASSERT_EQ(layer4->insData[1].lock(), getData("data5"));
    ASSERT_EQ(layer2->insData.size(), 3);
    ASSERT_EQ(layer2->insData[0].lock(), data2);
    ASSERT_EQ(layer2->insData[1].lock(), getData("data3"));
    ASSERT_EQ(layer2->insData[2].lock(), newData7);
}

TEST_F(RemoveLayerTests, canFindConstLayers) {
    getLayer("input1")->type = "Const";
    getLayer("layer2")->type = "Shape";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto constLayers = testTransformator->getConstLayers(sortedLayers);

    ASSERT_EQ(constLayers.size(), 2);
    auto begin = constLayers.begin();
    auto end = constLayers.end();
    ASSERT_FALSE(constLayers.at("input1"));
    ASSERT_FALSE(constLayers.at("layer2"));
}

TEST_F(RemoveLayerTests, canFindConstLayers2) {
    getLayer("input3")->type = "Const";
    getLayer("input2")->type = "Const";
    getLayer("layer2")->type = "Shape";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto constLayers = testTransformator->getConstLayers(sortedLayers);

    ASSERT_EQ(constLayers.size(), 4);
    ASSERT_FALSE(constLayers.at("input3"));
    ASSERT_FALSE(constLayers.at("layer2"));
    ASSERT_FALSE(constLayers.at("layer3"));
    ASSERT_FALSE(constLayers.at("input2"));
}

TEST_F(RemoveLayerTests, canFindConstLayers3) {
    getLayer("input3")->type = "Const";
    getLayer("layer2")->type = "Shape";
    getLayer("layer1")->type = "Shape";
    getLayer("layer4")->type = "Reshape";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto constLayers = testTransformator->getConstLayers(sortedLayers);

    ASSERT_EQ(constLayers.size(), 6);
    ASSERT_FALSE(constLayers.at("input3"));
    ASSERT_FALSE(constLayers.at("layer1"));
    ASSERT_TRUE(constLayers.at("layer2"));
    ASSERT_FALSE(constLayers.at("layer3"));
    ASSERT_FALSE(constLayers.at("layer4"));
    ASSERT_FALSE(constLayers.at("layer5"));
}

TEST_F(RemoveLayerTests, canFindShapeConstLayers) {
    getLayer("input3")->type = "Const";
    getLayer("layer2")->type = "Shape";
    getLayer("layer1")->type = "Shape";
    getLayer("layer6")->type = "Interp";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto constLayers = testTransformator->getConstLayers(sortedLayers);

    ASSERT_EQ(constLayers.size(), 6);
    ASSERT_TRUE(constLayers.at("input3"));
    ASSERT_TRUE(constLayers.at("layer1"));
    ASSERT_TRUE(constLayers.at("layer2"));
    ASSERT_TRUE(constLayers.at("layer3"));
    ASSERT_TRUE(constLayers.at("layer4"));
    ASSERT_TRUE(constLayers.at("layer5"));
}

TEST_F(RemoveLayerTests, canFindShapeConstLayers2) {
    getLayer("input3")->type = "Const";
    getLayer("input2")->type = "Const";
    getLayer("layer2")->type = "Shape";
    getLayer("layer1")->type = "Resample";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto constLayers = testTransformator->getConstLayers(sortedLayers);

    ASSERT_EQ(constLayers.size(), 4);
    ASSERT_FALSE(constLayers.at("input3"));
    ASSERT_FALSE(constLayers.at("layer2"));
    ASSERT_FALSE(constLayers.at("layer3"));
    ASSERT_FALSE(constLayers.at("input2"));
}

TEST_F(RemoveLayerTests, canTrimShapeInput) {
    std::vector<std::string> constLayers = {"input3", "layer3", "input2"};
    for (const auto& name : constLayers) {
        getLayer(name)->type = "Const";
    }
    getLayer("layer2")->type = "Shape";
    getLayer("layer1")->type = "Interp";
    getLayer("layer4")->type = "Reshape";
    getLayer("layer5")->type = "Reshape";
    auto layer1 = getLayer("layer1");
    auto layer4 = getLayer("layer4");
    auto layer5 = getLayer("layer5");

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto mapConstLayers = testTransformator->getConstLayers(sortedLayers);
    auto newLayers = testTransformator->foldConstSubgraphsInternal(mapConstLayers, {}, sortedLayers);
    testTransformator->trimShapeInputs(newLayers, sortedLayers);

    ASSERT_EQ(nullptr, net->getData("data5"));
    ASSERT_EQ(nullptr, net->getData("data2"));
    net->removeData("data5");
    net->removeData("data2");
    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 3);
    ASSERT_EQ(layer1->insData.size(), 1);
    ASSERT_EQ(layer1->insData[0].lock(), getData("data1"));
    ASSERT_EQ(layer4->insData.size(), 1);
    ASSERT_EQ(layer4->insData[0].lock(), getData("data4"));
    ASSERT_EQ(layer5->insData.size(), 2);
    ASSERT_EQ(layer5->insData[0].lock(), getData("data8"));
    ASSERT_EQ(layer5->insData[1].lock(), getData("data6"));
}

TEST_F(RemoveLayerTests, canTrimShapeInput2) {
    std::vector<std::string> constLayer_names = {"input3", "input2"};
    for (const auto& name : constLayer_names) {
        getLayer(name)->type = "Const";
    }
    auto layer1 = getLayer("layer1");
    auto layer2 = getLayer("layer2");
    layer1->type = "Resample";
    layer2->type = "StridedSlice";

    std::vector<InferenceEngine::CNNLayerPtr> constLayers;
    for (auto &const_input_name : constLayer_names)
        constLayers.push_back(getLayer(const_input_name));

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    testTransformator->trimShapeInputs(constLayers, sortedLayers);

    auto data6 = net->getData("data6");
    auto data2 = net->getData("data2");
    ASSERT_EQ(data2->getInputTo().size(), 1);
    ASSERT_EQ(data2->getInputTo().at(layer2->name), layer2);
    ASSERT_EQ(net->allLayers().size(), originalLayersNum);
    ASSERT_EQ(layer1->insData.size(), 1);
    ASSERT_EQ(layer1->insData[0].lock(), getData("data1"));
    ASSERT_EQ(layer2->insData.size(), 3);
    ASSERT_EQ(layer2->insData[0].lock(), getData("data2"));
    ASSERT_EQ(layer2->insData[1].lock(), getData("data3"));
    ASSERT_EQ(layer2->insData[2].lock(), getData("data7"));
}

TEST_F(RemoveLayerTests, notTrimFirstConstInput) {
    std::vector<std::string> testLayers = {"Interp", "Reshape", "Pad", "Gather", "Resample"};
    auto constLayer = getLayer("input4");
    constLayer->type = "Const";
    auto layer6 = getLayer("layer6");
    auto data10 = getData("data10");
    for (const auto& name: testLayers) {
        layer6->type = name;

        auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
        testTransformator->trimShapeInputs({constLayer}, sortedLayers);

        ASSERT_EQ(net->allLayers().size(), originalLayersNum);
        IE::CNNNetwork cnnNetwork(net);
        auto input4 = cnnNetwork.getLayerByName(constLayer->name.c_str());
        ASSERT_EQ(data10->getInputTo().size(), 1);
        ASSERT_EQ(data10->getCreatorLayer().lock(), input4);
        ASSERT_EQ(layer6->insData.size(), 2);
        ASSERT_EQ(layer6->insData[0].lock(), data10);
        ASSERT_EQ(layer6->insData[1].lock(), getData("data9"));
    }
}

TEST_F(RemoveLayerTests, canSaveConstForEltWise) {
    auto input2 = getLayer("input2");
    auto layer1 = getLayer("layer1");
    auto data2 = getData("data2");
    input2->type = "Const";
    layer1->type = "Eltwise";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    testTransformator->trimShapeInputs({input2}, sortedLayers);

    IE::CNNNetwork cnnNetwork(net);
    ASSERT_NO_THROW(input2 = cnnNetwork.getLayerByName(input2->name.c_str()));
    ASSERT_EQ(net->allLayers().size(), 10);
    ASSERT_EQ(layer1->insData.size(), 2);
    ASSERT_EQ(layer1->insData[1].lock(), data2);
    ASSERT_EQ(data2->getInputTo().size(), 2);
    ASSERT_EQ(data2->getInputTo().at(layer1->name), layer1);
    ASSERT_EQ(data2->getCreatorLayer().lock(), input2);
}

TEST_F(RemoveLayerTests, canSaveDataWithMultipleInputTo) {
    auto input3 = getLayer("input3");
    auto layer2 = getLayer("layer2");
    auto layer3 = getLayer("layer3");
    auto data3 = getData("data3");
    input3->type = "Const";
    layer2->type = "Reshape";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    testTransformator->trimShapeInputs({input3}, sortedLayers);

    IE::CNNNetwork cnnNetwork(net);
    ASSERT_NO_THROW(input3 = cnnNetwork.getLayerByName(input3->name.c_str()));
    ASSERT_EQ(net->allLayers().size(), originalLayersNum);
    ASSERT_EQ(layer2->insData.size(), 2);
    ASSERT_EQ(layer2->insData[0].lock(), getData("data2"));
    ASSERT_EQ(layer2->insData[1].lock(), getData("data7"));
    ASSERT_EQ(data3->getInputTo().size(), 1);
    ASSERT_EQ(data3->getInputTo().at(layer3->name), layer3);
    ASSERT_EQ(data3->getCreatorLayer().lock(), input3);
    ASSERT_EQ(layer3->insData.size(), 1);
    ASSERT_EQ(layer3->insData[0].lock(), data3);
}

TEST_F(RemoveLayerTests, canFoldConstSubgraphToConst) {
    std::vector<std::string> constLayers = {"input1", "input2", "input3"};
    std::vector<std::string> refNewLayers = {"layer5__data9__Const"};
    for (const auto& name : constLayers) {
        getLayer(name)->type = "Const";
    }
    getLayer("layer2")->type = "Shape";

    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    auto mapConstLayers = testTransformator->getConstLayers(sortedLayers);
    auto newLayers = testTransformator->foldConstSubgraphsInternal(mapConstLayers, {}, sortedLayers);

    std::vector<std::string> newLayer_names;
    for (auto layer : newLayers) newLayer_names.push_back(layer->name);

    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 7);
    ASSERT_EQ(newLayer_names, refNewLayers);
    IE::CNNNetwork cnnNetwork(net);
    auto newLayer = cnnNetwork.getLayerByName(refNewLayers[0].c_str());
    ASSERT_EQ(newLayer->type, "Const");
    ASSERT_EQ(newLayer->outData[0], getData("data9"));
}

TEST_F(RemoveLayerTests, canGetConstData) {
    std::vector<std::string> constLayers = {"input2", "input3", "layer3"};
    IE::BlobMap refBlobs = initConstLayers(constLayers);
    std::map<std::string, bool> mapConstLayers;
    for (const auto& it : constLayers) {
        mapConstLayers[it] = false;
    }
    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);

    auto actBlobs = testTransformator->getConstData(mapConstLayers, sortedLayers);

    ASSERT_EQ(actBlobs.size(), refBlobs.size());
    for (const auto& it: refBlobs) {
        ASSERT_EQ(it.second, actBlobs[it.first]);
    }
}

TEST_F(RemoveLayerTests, canGetConstDataForUnknownImpl) {
    initConstLayers({"input1", "input2", "input3"});
    {
        getLayer("layer1")->type = "UNKNOWN";
        getLayer("layer2")->type = "UNKNOWN";
        getLayer("layer3")->type = "Shape";
        getLayer("layer4")->type = "UNKNOWN";
        getLayer("layer5")->type = "Mul";
        getLayer("layer6")->type = "Reshape";
    }
    auto sortedLayers = IE::details::CNNNetSortTopologically(*net);
    IE::SizeVector refShape = {1, 1, 3};

    auto mapConstLayers = testTransformator->getConstLayers(sortedLayers);
    auto actBlobs = testTransformator->getConstData(mapConstLayers, sortedLayers);

    ASSERT_EQ(getData("data9")->getTensorDesc().getDims(), refShape);
}

TEST_F(RemoveLayerTests, canFoldConstSubgraphs) {
    IE::BlobMap refBlobs = initConstLayers({"input1", "input2", "input3"});
    std::vector<std::string> refNewLayers = {"layer5__data9__Const"};
    {   // TODO: method for marking layers
        getLayer("layer1")->type = "Mul";
        getLayer("layer2")->type = "Shape";
        getLayer("layer3")->type = "Power";
        getLayer("layer3")->params = {{"power", "1"},
                                      {"scale", "2"},
                                      {"shift", "-4"}};
        getLayer("layer4")->type = "Mul";
        getLayer("layer5")->type = "Mul";
    }
    float arr[] = {-2.f, 0.f, 54.f};
    auto ref5 = make_blob_with_precision(getData("data9")->getTensorDesc(), arr);

    IE::ConstTransformer transformator(net.get());
    transformator.foldConstSubgraphs();

    IE::CNNNetwork cnnNetwork(net);
    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 7);
    auto newLayer = cnnNetwork.getLayerByName(refNewLayers[0].c_str());
    auto actualBlob = newLayer->blobs["custom"];
    ASSERT_NE(actualBlob, nullptr);
    ASSERT_FALSE(actualBlob->buffer() == nullptr);
    TestsCommon::compare(*actualBlob, *ref5);
    ASSERT_EQ(newLayer->type, "Const");
}

TEST_F(RemoveLayerTests, canSkipConstCalculation) {
    IE::BlobMap refBlobs = initConstLayers({"input1", "input2", "input3"});
    getLayer("layer6")->type = "Reshape";

    IE::ConstTransformer transformator(net.get());
    transformator.foldConstSubgraphs();

    IE::CNNNetwork cnnNetwork(net);
    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 8);
}

TEST_F(RemoveLayerTests, canFoldConstWithUnknownImplForShapeDefiningLayers) {
    IE::BlobMap refBlobs = initConstLayers({"input1", "input2", "input3"});
    {
        getLayer("layer1")->type = "UNKNOWN";
        getLayer("layer2")->type = "UNKNOWN";
        getLayer("layer3")->type = "Shape";
        getLayer("layer4")->type = "Reshape";
        getLayer("layer5")->type = "Mul";
        getLayer("layer6")->type = "Reshape";
    }

    IE::ConstTransformer transformator(net.get());
    transformator.foldConstSubgraphs();

    IE::CNNNetwork cnnNetwork(net);
    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 8);
    ASSERT_EQ(getLayer("layer6")->insData.size(), 1);
}

TEST_F(RemoveLayerTests, throwErrorOnFoldWithUnknownImplForNotShapeDefiningLayers) {
    IE::BlobMap refBlobs = initConstLayers({"input1", "input2", "input3"});
    {
        getLayer("layer1")->type = "UNKNOWN";
        getLayer("layer2")->type = "Shape";
        getLayer("layer3")->type = "Shape";
        getLayer("layer4")->type = "Mul";
        getLayer("layer5")->type = "Mul";
        getLayer("layer6")->type = "Gather";
    }

    IE::ConstTransformer transformator(net.get());
    ASSERT_THROW(transformator.foldConstSubgraphs(), IE::details::InferenceEngineException);
}

TEST_F(RemoveLayerTests, canFullTrim) {
    IE::BlobMap refBlobs = initConstLayers({"input1", "input2", "input3"});
    auto layer6 = getLayer("layer6");
    {   // TODO: method for marking layers
        getLayer("layer1")->type = "Mul";
        getLayer("layer2")->type = "Shape";
        getLayer("layer3")->type = "Power";
        getLayer("layer3")->params = {{"power", "1"},
                                      {"scale", "2"},
                                      {"shift", "-4"}};
        getLayer("layer4")->type = "Mul";
        getLayer("layer5")->type = "Mul";
        layer6->type = "Reshape";
    }

    IE::ConstTransformer transformator(net.get());
    transformator.fullTrim();

    IE::CNNNetwork cnnNetwork(net);
    std::string newName = "layer5__data9__Const";
    ASSERT_THROW(cnnNetwork.getLayerByName(newName.c_str()), IE::NotFound);
    ASSERT_EQ(net->allLayers().size(), 2);
    ASSERT_EQ(layer6->insData.size(), 1);
    ASSERT_EQ(layer6->insData[0].lock(), getData("data10"));
}

TEST_F(AdvancedShapeInferTests, canFullTrimConstToReshape) {
    //
    //      I2-d2
    //          \
    //  I1-d1-Reshape-d3-L2-d4
    //
    net = netBuilder
            .data("data1", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{3, 1, 1}, IE::Layout::CHW))
            .data("data2", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{3}, IE::Layout::C))
            .data("data3", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .data("data4", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .layer<IE::CNNLayer>(IE::LayerParams{"input1", "Input", IE::Precision::I32})
            .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer1", "Reshape", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer2", "dummy", IE::Precision::FP32})
            .linkToData("input1", "data1")
            .linkToData("input2", "data2")
            .linkDataTo("data1", "layer1")
            .linkDataTo("data2", "layer1")
            .linkToData("layer1", "data3")
            .linkDataTo("data3", "layer2")
            .linkToData("layer2", "data4")
            .addInput("data1")
            .addInput("data2")
            .finalize();

    IE::BlobMap refBlobs = initConstLayers({"input2"});
    auto layer1 = getLayer("layer1");

    IE::ConstTransformer transformator(net.get());
    transformator.fullTrim();

    IE::CNNNetwork cnnNetwork(net);
    ASSERT_EQ(net->allLayers().size(), 3);
    ASSERT_EQ(layer1->insData.size(), 1);
    ASSERT_EQ(layer1->insData[0].lock(), getData("data1"));
}

TEST_F(AdvancedShapeInferTests, canReshape) {
    //
    // I2-d2-Shape
    //         \
    //         d3
    //          \
    //  I1-d1-Reshape-d4
    //
    net = netBuilder
            .data("data1", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{3, 1, 1}, IE::Layout::CHW))
            .data("data2", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .data("data3", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .data("data4", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .layer<IE::CNNLayer>(IE::LayerParams{"input1", "input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer1", "Reshape", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer2", "Shape", IE::Precision::FP32})
            .linkToData("input1", "data1")
            .linkToData("input2", "data2")
            .linkDataTo("data1", "layer1")
            .linkDataTo("data2", "layer2")
            .linkToData("layer2", "data3")
            .linkDataTo("data3", "layer1")
            .linkToData("layer1", "data4")
            .addInput("data1")
            .addInput("data2")
            .finalize();
    originalLayersNum = net->allLayers().size();
    IE::CNNNetwork cnnNetwork(net);
    IE::SizeVector newShape = {1, 3, 1};
    std::map<std::string, IE::SizeVector> inputShapes = {{"data2", newShape}};
    cnnNetwork.reshape(inputShapes);

    ASSERT_NO_THROW(cnnNetwork.getLayerByName("layer2"));
    ASSERT_EQ(getData("data3")->getTensorDesc().getDims(), IE::SizeVector{3});
    ASSERT_EQ(net->allLayers().size(), originalLayersNum);

    IE::ConstTransformer transformator(net.get());
    transformator.fullTrim();

    ASSERT_THROW(cnnNetwork.getLayerByName("layer2"), IE::NotFound);
    ASSERT_EQ(getData("data4")->getTensorDesc().getDims(), newShape);
    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 1);
}

TEST_F(AdvancedShapeInferTests, canReshape2) {
    //
    //                 I3-d3-Shape(L3)-d5
    //                                  \
    // I2-d2-Shape(L2)-d4-Power(L4)-d6-Mul(L5)-d7
    //                                          \
    //                                   I1-d1-Reshape(L1)-d8
    //
    net = netBuilder
            .data("data1", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .data("data2", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .data("data3", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .data("data4", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .data("data5", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .data("data6", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .data("data7", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .data("data8", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .layer<IE::CNNLayer>(IE::LayerParams{"input1", "input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"input3", "Input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer1", "Reshape", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer2", "Shape", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer3", "Shape", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer4", "Power", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer5", "Mul", IE::Precision::FP32})
            .linkToData("input1", "data1")
            .linkToData("input2", "data2")
            .linkToData("input3", "data3")

            .linkDataTo("data1", "layer1")
            .linkDataTo("data2", "layer2")
            .linkDataTo("data3", "layer3")

            .linkToData("layer2", "data4")
            .linkToData("layer3", "data5")

            .linkDataTo("data4", "layer4")

            .linkToData("layer4", "data6")

            .linkDataTo("data5", "layer5")
            .linkDataTo("data6", "layer5")

            .linkToData("layer5", "data7")

            .linkDataTo("data7", "layer1")

            .linkToData("layer1", "data8")

            .addInput("data1")
            .addInput("data2")
            .addInput("data3")
            .finalize();
    originalLayersNum = net->allLayers().size();
    IE::CNNNetwork cnnNetwork(net);
    IE::SizeVector newShape = {5, 9, 3};
    std::map<std::string, IE::SizeVector> inputShapes = {{"data1", {135}},
                                                         {"data2", {2, 1, 1}},
                                                         {"data3", {1, 3, 1}}};
    getLayer("layer4")->params = {{"power", "1"},
                                  {"scale", "2"},
                                  {"shift", "1"}};

    cnnNetwork.reshape(inputShapes);

    ASSERT_EQ(getData("data7")->getTensorDesc().getDims(), IE::SizeVector{3});
    ASSERT_EQ(net->allLayers().size(), originalLayersNum);

    IE::ConstTransformer transformator(net.get());
    transformator.fullTrim();

    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 4);
    ASSERT_EQ(getData("data8")->getTensorDesc().getDims(), newShape);
}

TEST_F(AdvancedShapeInferTests, canReshapeConst) {
    //
    //    Const-d2
    //           \
    // I1-d1-Reshape(L1)-d3
    //
    net = netBuilder
            .data("data1", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1 }, IE::Layout::C))
            .data("data2", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 3 }, IE::Layout::C))
            .data("data3", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{ 1, 1, 1 }, IE::Layout::CHW))
            .layer<IE::CNNLayer>(IE::LayerParams{"input1", "input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"const1", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer1", "Reshape", IE::Precision::FP32})
            .linkToData("input1", "data1")
            .linkToData("const1", "data2")
            .linkDataTo("data1", "layer1")
            .linkDataTo("data2", "layer1")
            .linkToData("layer1", "data3")
            .addInput("data1")
            .finalize();
    originalLayersNum = net->allLayers().size();
    IE::CNNNetwork cnnNetwork(net);
    initConstLayers({"const1"});
    IE::SizeVector newOutShape = {1, 2, 3};
    IE::SizeVector newInShape = {IE::details::product(newOutShape)};

    std::map<std::string, IE::SizeVector> inputShapes = {{"data1", newInShape}};

    cnnNetwork.reshape(inputShapes);

    ASSERT_EQ(net->allLayers().size(), originalLayersNum);

    IE::ConstTransformer transformator(net.get());
    transformator.fullTrim();

    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 1);
    ASSERT_EQ(getData("data1")->getTensorDesc().getDims(), newInShape);
    ASSERT_EQ(getData("data3")->getTensorDesc().getDims(), newOutShape);
}

TEST_F(AdvancedShapeInferTests, canReshapeCHWConst) {
    //
    //    Const-d1-Tile-d2
    //
    net = netBuilder
            .data("data1", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 3}, IE::Layout::CHW))
            .data("data2", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1, 1, 1}, IE::Layout::CHW))
            .layer<IE::CNNLayer>(IE::LayerParams{"const", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"tile", "Tile", IE::Precision::FP32})
            .linkToData("const", "data1")
            .linkDataTo("data1", "tile")
            .linkToData("tile", "data2")
            .addInput("data1")
            .finalize();
    getLayer("tile")->params = {{"axis",  "0"},
                                {"tiles", "2"}};
    originalLayersNum = net->allLayers().size();
    IE::CNNNetwork cnnNetwork(net);
    initConstLayers({"const"});

    cnnNetwork.reshape({});

    IE::SizeVector expectedDims = {2, 1, 3};
    ASSERT_EQ(getData("data2")->getTensorDesc().getDims(), expectedDims);
}

TEST_F(AdvancedShapeInferTests, canReshapeWithScalar) {
    //
    //   Scalar-d2
    //           \
    // I1-d1-Reshape(L1)-d3
    //
    net = netBuilder
            .data("data1", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .data("data2", IE::TensorDesc(IE::Precision::FP32, IE::Layout::SCALAR))
            .data("data3", IE::TensorDesc(IE::Precision::FP32, IE::SizeVector{1}, IE::Layout::C))
            .layer<IE::CNNLayer>(IE::LayerParams{"input1", "input", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"scalar", "dummy", IE::Precision::FP32})
            .layer<IE::CNNLayer>(IE::LayerParams{"layer1", "Reshape", IE::Precision::FP32})
            .linkToData("input1", "data1")
            .linkToData("scalar", "data2")
            .linkDataTo("data1", "layer1")
            .linkDataTo("data2", "layer1")
            .linkToData("layer1", "data3")
            .addInput("data1")
            .finalize();
    originalLayersNum = net->allLayers().size();
    IE::CNNNetwork cnnNetwork(net);
    initConstLayers({"scalar"});
    IE::SizeVector newOutShape = {1};
    IE::SizeVector newInShape = {IE::details::product(newOutShape)};

    std::map<std::string, IE::SizeVector> inputShapes = {{"data1", newInShape}};

    cnnNetwork.reshape(inputShapes);

    ASSERT_EQ(net->allLayers().size(), originalLayersNum);

    IE::ConstTransformer transformator(net.get());
    transformator.fullTrim();

    ASSERT_EQ(net->allLayers().size(), originalLayersNum - 1);
    ASSERT_EQ(getData("data1")->getTensorDesc().getDims(), newInShape);
    ASSERT_EQ(getData("data3")->getTensorDesc().getDims(), newOutShape);
}

TEST_F(AdvancedShapeInferTests, canFoldConstWithOneHot) {
    //   Const-d1-OneHot-d2
    //                     \
    //              I1-d3-Eltw(Sum)-d4
    auto testFunc = [&](IE::Precision precision) {
        netBuilder = NetBuilder();
        net = netBuilder
                .data("data1", IE::TensorDesc(precision, IE::SizeVector{2}, IE::Layout::C))
                .data("data2", IE::TensorDesc(precision, IE::SizeVector{10, 2}, IE::Layout::NC))
                .data("data3", IE::TensorDesc(precision, IE::SizeVector{10, 2}, IE::Layout::NC))
                .data("data4", IE::TensorDesc(precision, IE::SizeVector{10, 2}, IE::Layout::NC))
                .layer<IE::CNNLayer>(IE::LayerParams{"const", "dummy", precision})
                .layer<IE::CNNLayer>(IE::LayerParams{"oneHot", "OneHot", precision})
                .layer<IE::CNNLayer>(IE::LayerParams{"input", "input", precision})
                .layer<IE::CNNLayer>(IE::LayerParams{"eltwise", "Eltwise", precision})
                .linkToData("const", "data1")
                .linkDataTo("data1", "oneHot")
                .linkToData("oneHot", "data2")
                .linkDataTo("data2", "eltwise")
                .linkToData("input", "data3")
                .linkDataTo("data3", "eltwise")
                .linkToData("eltwise", "data4")
                .addInput("data3")
                .finalize();
        getLayer("oneHot")->params = {
                {"axis",      "-1"},
                {"depth",     "10"},
                {"off_value", "1.0"},
                {"on_value",  "1.0"}
        };
        getLayer("eltwise")->params = {
                {"operation", "sum"}
        };
        originalLayersNum = net->allLayers().size();

        IE::CNNNetwork cnnNetwork(net);
        initConstLayers({"const"});
        IE::ConstTransformer transformator(net.get());
        transformator.fullTrim();

        ASSERT_EQ(net->allLayers().size(), originalLayersNum - 1);
    };

    testFunc(IE::Precision::FP32);
    testFunc(IE::Precision::FP16);
    testFunc(IE::Precision::Q78);
    testFunc(IE::Precision::I16);
    testFunc(IE::Precision::U8);
    testFunc(IE::Precision::I8);
    testFunc(IE::Precision::U16);
    testFunc(IE::Precision::I32);
    testFunc(IE::Precision::I64);
}

TEST_F(AdvancedShapeInferTests, MulWithTensorConstInferTest) {

    auto testFunc = [&](IE::Precision precisionInData1, IE::Precision precisionInData2, IE::Precision precisionOutData) {

        netBuilder = NetBuilder();
        net = netBuilder
                .data("data1", IE::TensorDesc(precisionInData1, IE::SizeVector{2, 2}, IE::Layout::NC))
                .data("data2", IE::TensorDesc(precisionInData2, IE::SizeVector{2, 2}, IE::Layout::NC))
                .data("data3", IE::TensorDesc(precisionOutData, IE::SizeVector{2, 2}, IE::Layout::NC))
                .layer<IE::CNNLayer>(IE::LayerParams{"mulLayer", "Eltwise"})
                .layer<IE::CNNLayer>(IE::LayerParams{"input1", "Const", precisionInData1})
                .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Const", precisionInData2})
                .linkToData("input1", "data1")
                .linkToData("input2", "data2")
                .linkDataTo("data1", "mulLayer")
                .linkDataTo("data2", "mulLayer")
                .linkToData("mulLayer", "data3")
                .addInput("data1")
                .addInput("data2")
                .finalize();

        getLayer("mulLayer")->params = {
                {"operation", "mul"}
        };

        IE::CNNNetwork cnnNetwork(net);
        initConstLayersDiffPrec({"input1", "input2"});
        float ref[] = {4, 9, 16, 25};
        if (precisionOutData == IE::Precision::FP16) {
            for (int i = 0; i < 4; i++)
                ref[i] = IE::PrecisionUtils::f32tof16(ref[i]);
        }
        IE::ConstTransformer transformator(net.get());
        transformator.foldConstSubgraphs();
        switch(precisionOutData) {
            case IE::Precision::U8: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<uint8_t *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I32: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I64: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<long long int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP16: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<IE::ie_fp16 *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP32: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<float *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    };

    testFunc(IE::Precision::U8, IE::Precision::U8, IE::Precision::U8);
    testFunc(IE::Precision::U8, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::U8, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::U8, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::U8, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::U8, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::U8, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I32, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::U8, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::U8, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP16);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP16);
}


TEST_F(AdvancedShapeInferTests, MulWithScalarConstInferTest) {

    auto testFunc = [&](IE::Precision precisionInData1, IE::Precision precisionInData2, IE::Precision precisionOutData) {

        netBuilder = NetBuilder();
        net = netBuilder
                .data("data1", IE::TensorDesc(precisionInData1, IE::SizeVector{2, 2}, IE::Layout::NC))
                .data("data2", IE::TensorDesc(precisionInData2, IE::Layout::SCALAR))
                .data("data3", IE::TensorDesc(precisionOutData, IE::SizeVector{2, 2}, IE::Layout::NC))
                .layer<IE::CNNLayer>(IE::LayerParams{"mulLayer", "Eltwise"})
                .layer<IE::CNNLayer>(IE::LayerParams{"input1", "Const", precisionInData1})
                .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Const", precisionInData2})
                .linkToData("input1", "data1")
                .linkToData("input2", "data2")
                .linkDataTo("data1", "mulLayer")
                .linkDataTo("data2", "mulLayer")
                .linkToData("mulLayer", "data3")
                .addInput("data1")
                .addInput("data2")
                .finalize();

        getLayer("mulLayer")->params = {
                {"operation", "mul"}
        };

        IE::CNNNetwork cnnNetwork(net);
        initConstLayersDiffPrec({"input1", "input2"});
        float ref[] = {4, 6, 8, 10};
        if (precisionOutData == IE::Precision::FP16) {
            for (int i = 0; i < 4; i++)
                ref[i] = IE::PrecisionUtils::f32tof16(ref[i]);
        }
        IE::ConstTransformer transformator(net.get());
        transformator.foldConstSubgraphs();
        switch(precisionOutData) {
            case IE::Precision::U8: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<uint8_t *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I32: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I64: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<long long int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP16: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<IE::ie_fp16 *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP32: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<float *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    };

    testFunc(IE::Precision::U8, IE::Precision::U8, IE::Precision::U8);
    testFunc(IE::Precision::U8, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::U8, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::U8, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::U8, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::U8, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::U8, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I32, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::U8, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::U8, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP16);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP16);
}

TEST_F(AdvancedShapeInferTests, AddWithScalarConstInferTest) {

    auto testFunc = [&](IE::Precision precisionInData1, IE::Precision precisionInData2, IE::Precision precisionOutData) {

        netBuilder = NetBuilder();
        net = netBuilder
                .data("data1", IE::TensorDesc(precisionInData1, IE::SizeVector{2, 2}, IE::Layout::NC))
                .data("data2", IE::TensorDesc(precisionInData2, IE::Layout::SCALAR))
                .data("data3", IE::TensorDesc(precisionOutData, IE::SizeVector{2, 2},IE::Layout::NC))
                .layer<IE::CNNLayer>(IE::LayerParams{"addLayer", "Eltwise"})
                .layer<IE::CNNLayer>(IE::LayerParams{"input1", "Const", precisionInData1})
                .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Const", precisionInData2})
                .linkToData("input1", "data1")
                .linkToData("input2", "data2")
                .linkDataTo("data1", "addLayer")
                .linkDataTo("data2", "addLayer")
                .linkToData("addLayer", "data3")
                .addInput("data1")
                .addInput("data2")
                .finalize();

        getLayer("addLayer")->params = {
                {"operation", "sum"}
        };

        IE::CNNNetwork cnnNetwork(net);
        initConstLayersDiffPrec({"input1", "input2"});
        float ref[] = {4, 5, 6, 7};
        if (precisionOutData == IE::Precision::FP16) {
            for (int i = 0; i < 4; i++)
                ref[i] = IE::PrecisionUtils::f32tof16(ref[i]);
        }
        IE::ConstTransformer transformator(net.get());
        transformator.foldConstSubgraphs();
        switch(precisionOutData) {
            case IE::Precision::U8: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<uint8_t *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I32: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I64: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<long long int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP16: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<IE::ie_fp16 *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP32: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<float *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    };

    testFunc(IE::Precision::U8, IE::Precision::U8, IE::Precision::U8);
    testFunc(IE::Precision::U8, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::U8, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::U8, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::U8, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::U8, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::U8, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I32, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::U8, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::U8, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP16);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP16);
}

TEST_F(AdvancedShapeInferTests, AddWithTensorConstInferTest) {

    auto testFunc = [&](IE::Precision precisionInData1, IE::Precision precisionInData2, IE::Precision precisionOutData) {

        netBuilder = NetBuilder();
        net = netBuilder
                .data("data1", IE::TensorDesc(precisionInData1, IE::SizeVector{2, 2}, IE::Layout::NC))
                .data("data2", IE::TensorDesc(precisionInData2, IE::SizeVector{2, 2}, IE::Layout::NC))
                .data("data3", IE::TensorDesc(precisionOutData, IE::SizeVector{2, 2}, IE::Layout::NC))
                .layer<IE::CNNLayer>(IE::LayerParams{"addLayer", "Eltwise"})
                .layer<IE::CNNLayer>(IE::LayerParams{"input1", "Const", precisionInData1})
                .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Const", precisionInData2})
                .linkToData("input1", "data1")
                .linkToData("input2", "data2")
                .linkDataTo("data1", "addLayer")
                .linkDataTo("data2", "addLayer")
                .linkToData("addLayer", "data3")
                .addInput("data1")
                .addInput("data2")
                .finalize();

        getLayer("addLayer")->params = {
                {"operation", "sum"}
        };

        IE::CNNNetwork cnnNetwork(net);
        initConstLayersDiffPrec({"input1", "input2"});
        float ref[] = {4, 6, 8, 10};
        if (precisionOutData == IE::Precision::FP16) {
            for (int i = 0; i < 4; i++)
                ref[i] = IE::PrecisionUtils::f32tof16(ref[i]);
        }
        IE::ConstTransformer transformator(net.get());
        transformator.foldConstSubgraphs();
        switch(precisionOutData) {
            case IE::Precision::U8: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<uint8_t *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I32: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I64: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<long long int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP16: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<IE::ie_fp16 *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP32: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<float *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    };

    testFunc(IE::Precision::U8, IE::Precision::U8, IE::Precision::U8);
    testFunc(IE::Precision::U8, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::U8, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::U8, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::U8, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::U8, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::U8, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I32, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::U8, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::U8, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP16);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP16);
}

TEST_F(AdvancedShapeInferTests, AddWithBroadcastingConstInferTest) {

    auto testFunc = [&](IE::Precision precisionInData1, IE::Precision precisionInData2, IE::Precision precisionOutData) {

        netBuilder = NetBuilder();
        net = netBuilder
                .data("data1", IE::TensorDesc(precisionInData1, IE::SizeVector{2, 2}, IE::Layout::NC))
                .data("data2", IE::TensorDesc(precisionInData2, IE::SizeVector{2, 1}, IE::Layout::NC))
                .data("data3", IE::TensorDesc(precisionOutData, IE::SizeVector{2, 2}, IE::Layout::NC))
                .layer<IE::CNNLayer>(IE::LayerParams{"addLayer", "Eltwise"})
                .layer<IE::CNNLayer>(IE::LayerParams{"input1", "Const", precisionInData1})
                .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Const", precisionInData2})
                .linkToData("input1", "data1")
                .linkToData("input2", "data2")
                .linkDataTo("data1", "addLayer")
                .linkDataTo("data2", "addLayer")
                .linkToData("addLayer", "data3")
                .addInput("data1")
                .addInput("data2")
                .finalize();

        getLayer("addLayer")->params = {
                {"operation", "sum"}
        };

        IE::CNNNetwork cnnNetwork(net);
        initConstLayersDiffPrec({"input1", "input2"});
        float ref[] = {4, 5, 7, 8};
        if (precisionOutData == IE::Precision::FP16) {
            for (int i = 0; i < 4; i++)
                ref[i] = IE::PrecisionUtils::f32tof16(ref[i]);
        }
        IE::ConstTransformer transformator(net.get());
        transformator.foldConstSubgraphs();
        switch(precisionOutData) {
            case IE::Precision::U8: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<uint8_t *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I32: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I64: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<long long int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP16: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<IE::ie_fp16 *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP32: {
                auto *l = cnnNetwork.getLayerByName("addLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<float *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    };

    testFunc(IE::Precision::U8, IE::Precision::U8, IE::Precision::U8);
    testFunc(IE::Precision::U8, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::U8, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::U8, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::U8, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::U8, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::U8, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I32, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::U8, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::U8, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP16);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP16);
}

TEST_F(AdvancedShapeInferTests, MulWithBroadcastingConstInferTest) {

    auto testFunc = [&](IE::Precision precisionInData1, IE::Precision precisionInData2, IE::Precision precisionOutData) {

        netBuilder = NetBuilder();
        net = netBuilder
                .data("data1", IE::TensorDesc(precisionInData1, IE::SizeVector{ 2, 2 }, IE::Layout::NC))
                .data("data2", IE::TensorDesc(precisionInData2, IE::SizeVector{ 2, 1 }, IE::Layout::NC))
                .data("data3", IE::TensorDesc(precisionOutData, IE::SizeVector{ 2, 2 }, IE::Layout::NC))
                .layer<IE::CNNLayer>(IE::LayerParams{"mulLayer", "Eltwise"})
                .layer<IE::CNNLayer>(IE::LayerParams{"input1", "Const", precisionInData1})
                .layer<IE::CNNLayer>(IE::LayerParams{"input2", "Const", precisionInData2})
                .linkToData("input1", "data1")
                .linkToData("input2", "data2")
                .linkDataTo("data1", "mulLayer")
                .linkDataTo("data2", "mulLayer")
                .linkToData("mulLayer", "data3")
                .addInput("data1")
                .addInput("data2")
                .finalize();

        getLayer("mulLayer")->params = {
                {"operation", "mul"}
        };

        IE::CNNNetwork cnnNetwork(net);
        initConstLayersDiffPrec({"input1", "input2"});
        float ref[] = {4, 6, 12, 15};
        if (precisionOutData == IE::Precision::FP16) {
            for (int i = 0; i < 4; i++)
                ref[i] = IE::PrecisionUtils::f32tof16(ref[i]);
        }
        IE::ConstTransformer transformator(net.get());
        transformator.foldConstSubgraphs();
        switch(precisionOutData) {
            case IE::Precision::U8: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<uint8_t *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I32: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::I64: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<long long int *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP16: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<IE::ie_fp16 *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            case IE::Precision::FP32: {
                auto *l = cnnNetwork.getLayerByName("mulLayer__data3__Const").get()->blobs.at("custom")->cbuffer().as<float *>();
                ASSERT_EQ(l[0], ref[0]);
                ASSERT_EQ(l[1], ref[1]);
                ASSERT_EQ(l[2], ref[2]);
                ASSERT_EQ(l[3], ref[3]);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    };

    testFunc(IE::Precision::U8, IE::Precision::U8, IE::Precision::U8);
    testFunc(IE::Precision::U8, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::U8, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::U8, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::U8, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::U8, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I32, IE::Precision::I32);
    testFunc(IE::Precision::I32, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::U8, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I32, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::I64, IE::Precision::I64);
    testFunc(IE::Precision::I64, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::I64, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::U8, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP16, IE::Precision::FP16);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::U8, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I32, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::I64, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP32);
    testFunc(IE::Precision::FP32, IE::Precision::FP32, IE::Precision::FP32);
    testFunc(IE::Precision::FP16, IE::Precision::FP32, IE::Precision::FP16);
    testFunc(IE::Precision::FP32, IE::Precision::FP16, IE::Precision::FP16);
}
