// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <shape_infer/built-in/ie_built_in_holder.hpp>

using namespace InferenceEngine;

class MockIShapeInferImpl : public IShapeInferImpl {
public:
    using Ptr = std::shared_ptr<MockIShapeInferImpl>;

    MOCK_QUALIFIED_METHOD5(inferShapes, noexcept, StatusCode(
            const std::vector<Blob::CPtr> &, const std::map<std::string, std::string>&, const std::map<std::string, Blob::Ptr>&, std::vector<SizeVector> &, ResponseDesc *));

};

