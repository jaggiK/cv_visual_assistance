// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <ie_version.hpp>
#include <cnn_network_impl.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>

#include <mock_icnn_network.hpp>
#include <mock_iexecutable_network.hpp>
#include <mock_not_empty_icnn_network.hpp>
#include <cpp_interfaces/mock_plugin_impl.hpp>
#include <cpp_interfaces/impl/mock_inference_plugin_internal.hpp>
#include <cpp_interfaces/impl/mock_executable_thread_safe_default.hpp>
#include <cpp_interfaces/interface/mock_iinfer_request_internal.hpp>
#include <mock_iasync_infer_request.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class InferenceEnginePluginInternalTest : public ::testing::Test {
protected:
    IE_SUPPRESS_DEPRECATED_START
    shared_ptr<IInferencePlugin> plugin;
    IE_SUPPRESS_DEPRECATED_END
    shared_ptr<MockInferencePluginInternal> mock_plugin_impl;
    shared_ptr<MockExecutableNetworkInternal> mockExeNetworkInternal;
    shared_ptr<MockExecutableNetworkThreadSafe> mockExeNetworkTS;
    shared_ptr<MockInferRequestInternal> mockInferRequestInternal;
    MockNotEmptyICNNNetwork mockNotEmptyNet;
    std::string pluginId;

    ResponseDesc dsc;
    StatusCode sts;

    virtual void TearDown() {
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mock_plugin_impl.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetworkInternal.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockExeNetworkTS.get()));
        EXPECT_TRUE(Mock::VerifyAndClearExpectations(mockInferRequestInternal.get()));
    }

    virtual void SetUp() {
        pluginId = "TEST";
        mock_plugin_impl.reset(new MockInferencePluginInternal());
        mock_plugin_impl->SetName(pluginId);
        plugin = details::shared_from_irelease(make_ie_compatible_plugin({{2, 1}, "test", "version"}, mock_plugin_impl));
        mockExeNetworkInternal = make_shared<MockExecutableNetworkInternal>();
        mockExeNetworkInternal->SetPointerToPluginInternal(mock_plugin_impl);
    }

    void getInferRequestWithMockImplInside(IInferRequest::Ptr &request) {
        IExecutableNetwork::Ptr exeNetwork;
        InputsDataMap inputsInfo;
        mockNotEmptyNet.getInputsInfo(inputsInfo);
        OutputsDataMap outputsInfo;
        mockNotEmptyNet.getOutputsInfo(outputsInfo);
        mockInferRequestInternal = make_shared<MockInferRequestInternal>(inputsInfo, outputsInfo);
        mockExeNetworkTS = make_shared<MockExecutableNetworkThreadSafe>();
        EXPECT_CALL(*mock_plugin_impl.get(), LoadExeNetworkImpl(_, _, _)).WillOnce(Return(mockExeNetworkTS));
        EXPECT_CALL(*mockExeNetworkTS.get(), CreateInferRequestImpl(_, _)).WillOnce(Return(mockInferRequestInternal));
        sts = plugin->LoadNetwork(exeNetwork, mockNotEmptyNet, {}, &dsc);
        ASSERT_EQ((int) StatusCode::OK, sts) << dsc.msg;
        ASSERT_NE(exeNetwork, nullptr) << dsc.msg;
        sts = exeNetwork->CreateInferRequest(request, &dsc);
        ASSERT_EQ((int) StatusCode::OK, sts) << dsc.msg;
    }
};

MATCHER_P(blob_in_map_pointer_is_same, ref_blob, "") {
    auto a = arg.begin()->second.get();
    return (float *) (arg.begin()->second->buffer()) == (float *) (ref_blob->buffer());
}

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithInCorrectName) {
    Blob::Ptr inBlob = make_shared_blob<float>({ Precision::FP32, {1, 1, 1, 1}, NCHW });
    inBlob->allocate();
    string inputName = "not_input";
    std::string refError = NOT_FOUND_str + "Failed to find input or output with name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetBlobWithNullPtr) {
    Blob::Ptr inBlob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });
    inBlob->allocate();
    string inputName = "not_input";
    std::string refError = NOT_FOUND_str + "Failed to set blob with empty name";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(nullptr, inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetNullPtr) {
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = NOT_ALLOCATED_str + "Failed to set empty blob with name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    Blob::Ptr inBlob = nullptr;

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetEmptyBlob) {
    Blob::Ptr inBlob;
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = NOT_ALLOCATED_str + "Failed to set empty blob with name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), inBlob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, failToSetNotAllocatedBlob) {
    string inputName = MockNotEmptyICNNNetwork::INPUT_BLOB_NAME;
    std::string refError = "Input data was not allocated. Input name: \'" + inputName + "\'";
    IInferRequest::Ptr inferRequest;
    getInferRequestWithMockImplInside(inferRequest);
    Blob::Ptr blob = make_shared_blob<float>({ Precision::FP32, {}, NCHW });

    ASSERT_NO_THROW(sts = inferRequest->SetBlob(inputName.c_str(), blob, &dsc));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    dsc.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, dsc.msg);
}

TEST_F(InferenceEnginePluginInternalTest, executableNetworkInternalExportsMagicAndName) {
    std::stringstream strm;
    ASSERT_NO_THROW(mockExeNetworkInternal->WrapOstreamExport(strm));
    ExportMagic actualMagic = {};
    strm.read(actualMagic.data(), actualMagic.size());
    ASSERT_EQ(exportMagic, actualMagic);
    std::string pluginName;
    std::getline(strm, pluginName);
    ASSERT_EQ(pluginId, pluginName);
    std::string exportedString;
    std::getline(strm, exportedString);
    ASSERT_EQ(mockExeNetworkInternal->exportString, exportedString);
}

TEST_F(InferenceEnginePluginInternalTest, pluginInternalEraseMagicAndNameWhenImports) {
    std::stringstream strm;
    ASSERT_NO_THROW(mockExeNetworkInternal->WrapOstreamExport(strm));
    ASSERT_NO_THROW(mock_plugin_impl->ImportNetwork(strm, {}));
    ASSERT_EQ(mockExeNetworkInternal->exportString, mock_plugin_impl->importedString);
    mock_plugin_impl->importedString = {};
}
