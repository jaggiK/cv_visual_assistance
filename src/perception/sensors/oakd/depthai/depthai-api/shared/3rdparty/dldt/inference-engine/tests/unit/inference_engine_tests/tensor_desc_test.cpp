// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_layouts.h>
#include <ie_blob.h>
#include <gtest/gtest.h>
#include <random>
#include <chrono>

#include "mock_allocator.hpp"

#include <cpp/ie_cnn_net_reader.h>
#include <gmock/gmock-spec-builders.h>

#ifdef WIN32
#define UNUSED
#else
#define UNUSED  __attribute__((unused))
#endif

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

class TensorDescTests: public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

public:

};

TEST_F(TensorDescTests, CreateBlobWithIncorrectLayout) {
    ASSERT_THROW(make_shared_blob<float>({ Precision::FP32, {1, 3, 32}, Layout::NC }), details::InferenceEngineException);
}

TEST_F(TensorDescTests, CreateBlockedBlobNCHW) {
    TensorDesc desc(Precision::FP32, {1, 4, 2, 1}, {{1, 2, 2, 1, 2}, {0, 1, 2, 3, 1}});
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Blob::Ptr blockedBlob = make_shared_blob<float>(desc, data);
    Blob::Ptr nchwBlob = make_shared_blob<float>({Precision::FP32, {1, 4, 2, 1}, Layout::NCHW}, data);
    ASSERT_NE(blockedBlob->getTensorDesc().offset(5), nchwBlob->getTensorDesc().offset(5));
    ASSERT_EQ(6, blockedBlob->getTensorDesc().offset(5));
    ASSERT_EQ(5, nchwBlob->getTensorDesc().offset(5));
    ASSERT_EQ(Layout::NCHW, nchwBlob->getTensorDesc().getLayout());
    ASSERT_EQ(Layout::BLOCKED, blockedBlob->getTensorDesc().getLayout());
}

TEST_F(TensorDescTests, CreateBlockedBlobNCDHW) {
    TensorDesc desc(Precision::FP32, {1, 4, 2, 2, 1}, {{1, 2, 2, 2, 1, 2}, {0, 1, 2, 3, 4, 1}});
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Blob::Ptr blockedBlob = make_shared_blob<float>(desc, data);
    Blob::Ptr ncdhwBlob = make_shared_blob<float>({Precision::FP32, {1, 4, 2, 2, 1}, Layout::NCDHW}, data);
    ASSERT_NE(blockedBlob->getTensorDesc().offset(6), ncdhwBlob->getTensorDesc().offset(6));
    ASSERT_EQ(5, blockedBlob->getTensorDesc().offset(6));
    ASSERT_EQ(6, ncdhwBlob->getTensorDesc().offset(6));
    ASSERT_EQ(Layout::NCDHW, ncdhwBlob->getTensorDesc().getLayout());
    ASSERT_EQ(Layout::BLOCKED, blockedBlob->getTensorDesc().getLayout());
}

TEST_F(TensorDescTests, CompareNHWCandNCHWLayouts) {
    TensorDesc descNCHW(Precision::FP32, {1, 3, 4, 2}, Layout::NCHW);
    TensorDesc descNHWC(Precision::FP32, {1, 3, 4, 2}, Layout::NHWC);
    SizeVector nchw = {0, 1, 2, 3};
    SizeVector nhwc = {0, 2, 3, 1};

    ASSERT_NE(descNCHW, descNHWC);
    ASSERT_NE(descNCHW.getBlockingDesc(), descNHWC.getBlockingDesc());
    ASSERT_NE(descNCHW.getBlockingDesc().getOrder(), descNHWC.getBlockingDesc().getOrder());
    ASSERT_EQ(descNCHW.getBlockingDesc().getOrder(), nchw);
    ASSERT_EQ(descNHWC.getBlockingDesc().getOrder(), nhwc);
}

TEST_F(TensorDescTests, CompareNDHWCandNCDHWLayouts) {
    TensorDesc descNCDHW(Precision::FP32, {1, 3, 4, 4, 2}, Layout::NCDHW);
    TensorDesc descNDHWC(Precision::FP32, {1, 3, 4, 4, 2}, Layout::NDHWC);
    SizeVector ncdhw = {0, 1, 2, 3, 4};
    SizeVector ndhwc = {0, 2, 3, 4, 1};

    ASSERT_NE(descNCDHW, descNDHWC);
    ASSERT_NE(descNCDHW.getBlockingDesc(), descNDHWC.getBlockingDesc());
    ASSERT_NE(descNCDHW.getBlockingDesc().getOrder(), descNDHWC.getBlockingDesc().getOrder());
    ASSERT_EQ(descNCDHW.getBlockingDesc().getOrder(), ncdhw);
    ASSERT_EQ(descNDHWC.getBlockingDesc().getOrder(), ndhwc);
}
