// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cnn_network_ngraph_impl.hpp>
#include "tests_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <cpp/ie_cnn_net_reader.h>
#include <cpp/ie_cnn_network.h>
#include <ie_util_internal.hpp>
#include <ie_parameter.hpp>
#include <ie_core.hpp>

#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/result.hpp>

using namespace testing;
using namespace InferenceEngine;

class CNNNGraphImplTests : public TestsCommon {};

TEST_F(CNNNGraphImplTests, TestConvertNetwork) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    auto cnnRefNet = cnnNet.getCNNNetwork();
    cnnNet.convertToCNNNetworkImpl();

    ASSERT_NE(cnnRefNet, cnnNet.getCNNNetwork());
}

TEST_F(CNNNGraphImplTests, TestGetOutputAfterConvertNetwork) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    cnnNet.begin();
    cnnNet.addOutput(testLayerName);

    InferenceEngine::OutputsDataMap outs = cnnNet.getOutputsInfo();
    ASSERT_EQ(2, outs.size());
}

TEST_F(CNNNGraphImplTests, TestSetBatch) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::details::CNNNetworkNGraphImpl cnnNet(ngraph);
    ASSERT_EQ(1, cnnNet.getBatchSize());
    ASSERT_EQ(OK, cnnNet.setBatchSize(2, nullptr));
    ASSERT_EQ(2, cnnNet.getBatchSize());
    auto cnnRefNet = cnnNet.getCNNNetwork();

    cnnNet.convertToCNNNetworkImpl();

    ASSERT_EQ(2, cnnNet.getBatchSize());
    ASSERT_EQ(2, cnnNet.getCNNNetwork()->getBatchSize());

    auto cnnNet2 = cnnNet.cloneNGraphImpl();

    ASSERT_EQ(2, cnnNet2->getBatchSize());
    ASSERT_EQ(2, cnnNet2->getCNNNetwork()->getBatchSize());
    ASSERT_NE(cnnRefNet, cnnNet2->getCNNNetwork());
}

TEST_F(CNNNGraphImplTests, TestSaveAffinity) {
    const std::string testAffinity = "testAffinity";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto& rtInfo = relu->get_rt_info();
        InferenceEngine::Parameter rt(testAffinity);
        rtInfo["affinity"] = rt.asVariant();
        relu->set_friendly_name("testReLU");
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    auto cnnLayer = cnnNet.getLayerByName("testReLU");
    ASSERT_NE(nullptr, cnnLayer);
    ASSERT_EQ(cnnLayer->affinity, testAffinity);
}

TEST_F(CNNNGraphImplTests, TestAddOutput) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(4, cnnNet.layerCount());

    cnnNet.addOutput(testLayerName);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());
    auto outputs = cnnNet.getOutputsInfo();
    ASSERT_EQ(2, outputs.size());
    ASSERT_TRUE(outputs.find("relu2") != outputs.end());
    ASSERT_TRUE(outputs.find(testLayerName) != outputs.end());
}

TEST_F(CNNNGraphImplTests, TestAddOutputFromConvertedNetwork) {
    const std::string testLayerName = "testReLU";
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name(testLayerName);
        auto relu2 = std::make_shared<ngraph::op::Relu>(relu);
        relu2->set_friendly_name("relu2");
        auto result = std::make_shared<ngraph::op::Result>(relu2);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(4, cnnNet.layerCount());

    cnnNet.addOutput(testLayerName);
    ASSERT_NE(nullptr, cnnNet.getFunction());
    ASSERT_EQ(5, cnnNet.layerCount());
    cnnNet.begin();
    auto outputs = cnnNet.getOutputsInfo();
    ASSERT_EQ(2, outputs.size());
    ASSERT_TRUE(outputs.find("relu2") != outputs.end());
    ASSERT_TRUE(outputs.find(testLayerName) != outputs.end());
}

TEST_F(CNNNGraphImplTests, ConstantAsInternalAndExternalLayer) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto constant = ngraph::op::Constant::create(ngraph::element::Type_t::f32, {1}, {2});
        auto prelu = std::make_shared<ngraph::op::PRelu>(param, constant);
        auto add = std::make_shared<ngraph::op::v1::Maximum>(prelu, constant);
        auto result = std::make_shared<ngraph::op::Result>(add);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    cnnNet.begin();
    ASSERT_EQ(4, cnnNet.layerCount());
}

TEST_F(CNNNGraphImplTests, SavePrimitivesPriority) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32" PrimitivesPriority="cpu:avx2"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
        std::string plugins_path;
#ifndef _WIN32
        plugins_path = "lib/";
#endif
        plugins_path += "plugins.xml";
        const Core ie(testing::FileUtils::makePath(getIELibraryPath(), plugins_path));
        Blob::Ptr weights;

        auto network = ie.ReadNetwork(model, weights);
        auto nGraph = network.getFunction();
        ASSERT_TRUE(nGraph);
        auto rt_info = nGraph->get_parameters()[0]->get_rt_info();
        ASSERT_NE(rt_info.find("PrimitivesPriority"), rt_info.end());
        Parameter param(rt_info["PrimitivesPriority"]);
        ASSERT_EQ("cpu:avx2", param.as<std::string>());

        auto inputInfo = network.getInputsInfo();
        auto cnnLayer = inputInfo.begin()->second->getInputData()->getCreatorLayer().lock();
        ASSERT_TRUE(cnnLayer);
        ASSERT_NE(cnnLayer->params.find("PrimitivesPriority"), cnnLayer->params.end());
        ASSERT_EQ("cpu:avx2", cnnLayer->params["PrimitivesPriority"]);
}

TEST_F(CNNNGraphImplTests, ReadFromCNNNetReader) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    CNNNetReader reader;
    reader.ReadNetwork(model.data(), model.length());
    ASSERT_TRUE(reader.isParseSuccess());
    reader.SetWeights(nullptr);
    CNNNetwork network = reader.getNetwork();
    network.begin();
    ASSERT_EQ(2, network.layerCount());
}

TEST_F(CNNNGraphImplTests, CanChangeInputPrecision) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto inputsInfo = cnnNet.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getPrecision(), Precision::FP32)
                << "FP32 is default presision";
    }
    {
        SCOPED_TRACE("Manually set input precision");

        const auto inputsInfo = cnnNet.getInputsInfo();

        inputsInfo.at("input")->setPrecision(Precision::FP16);
    }
    {
        SCOPED_TRACE("Convert to old format");

        cnnNet.begin();
    }
    {
        SCOPED_TRACE("After conversion");

        const auto inputsInfo = cnnNet.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getPrecision(), Precision::FP16)
                << "Manually set presision should be left unchanged";
    }
}

TEST_F(CNNNGraphImplTests, CanChangeInputLayout) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto inputsInfo = cnnNet.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getLayout(), Layout::NCHW)
                << "NCHW is default layout";
    }
    {
        SCOPED_TRACE("Manually set input layout");

        const auto inputsInfo = cnnNet.getInputsInfo();

        inputsInfo.at("input")->setLayout(Layout::NHWC);
    }
    {
        SCOPED_TRACE("Convert to old format");

        cnnNet.begin();
    }
    {
        SCOPED_TRACE("After conversion");

        const auto inputsInfo = cnnNet.getInputsInfo();

        ASSERT_EQ(inputsInfo.at("input")->getLayout(), Layout::NHWC)
                << "Manually set layout should be left unchanged";
    }
}

TEST_F(CNNNGraphImplTests, CanChangeOutputPrecision) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getPrecision(), Precision::FP32)
                << "FP32 is default presision";
    }
    {
        SCOPED_TRACE("Manually set output precision");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        outputsInfo.at("output")->setPrecision(Precision::FP16);
    }
    {
        SCOPED_TRACE("Convert to old format");

        cnnNet.begin();
    }
    {
        SCOPED_TRACE("After conversion");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getPrecision(), Precision::FP16)
                << "Manually set presision should be left unchanged";
    }
}

TEST_F(CNNNGraphImplTests, CanChangeOutputLayout) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 16, 16});
        ngraph::element::Type type(ngraph::element::Type_t::f16);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("input");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("output");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};
        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    InferenceEngine::CNNNetwork cnnNet(ngraph);
    {
        SCOPED_TRACE("After ctor");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getLayout(), Layout::NCHW)
                << "NCHW is default layout";
    }
    {
        SCOPED_TRACE("Manually set output layout");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        outputsInfo.at("output")->setLayout(Layout::NHWC);
    }
    {
        SCOPED_TRACE("Convert to old format");

        cnnNet.begin();
    }
    {
        SCOPED_TRACE("After conversion");

        const auto outputsInfo = cnnNet.getOutputsInfo();

        ASSERT_EQ(outputsInfo.at("output")->getLayout(), Layout::NHWC)
                << "Manually set layout should be left unchanged";
    }
}
