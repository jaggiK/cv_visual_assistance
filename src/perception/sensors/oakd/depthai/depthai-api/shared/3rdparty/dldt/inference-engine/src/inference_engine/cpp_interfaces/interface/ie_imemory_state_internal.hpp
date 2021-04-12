// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

#include <memory>
#include <string>

namespace InferenceEngine {
/**
 * @brief minimal interface for memory state implementation
 */
class IMemoryStateInternal {
public:
    using Ptr = std::shared_ptr<IMemoryStateInternal>;

    virtual ~IMemoryStateInternal() = default;
    virtual std::string GetName() const = 0;
    virtual void Reset() = 0;
    virtual void SetState(Blob::Ptr newState) = 0;
    virtual Blob::CPtr GetLastState() const = 0;
};

}  // namespace InferenceEngine