// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_common.h>

#include <map>
#include <memory>
#include <string>

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in InferRequestBase forwarding mechanism
 */
class IInferRequestInternal {
public:
    typedef std::shared_ptr<IInferRequestInternal> Ptr;

    virtual ~IInferRequestInternal() = default;

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of IInferRequest while request is ongoing (running or waiting in queue)
     */
    virtual void Infer() = 0;

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  Note: not all plugins may provide meaningful data
     *  @param perfMap - a map of layer names to profiling information for that layer.
     */
    virtual void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap) const = 0;

    /**
     * @brief Set input/output data to infer
     * @note: Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     */
    virtual void SetBlob(const char* name, const Blob::Ptr& data) = 0;

    /**
     * @brief Get input/output data to infer
     * @note: Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     */
    virtual void GetBlob(const char* name, Blob::Ptr& data) = 0;

    /**
     * @brief Sets pre-process for input data
     * @param name Name of input blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     * @param info Preprocess info for blob.
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     */
    virtual void SetBlob(const char* name, const Blob::Ptr& data, const PreProcessInfo& info) = 0;

    /**
     * @brief Gets pre-process for input data
     * @param name Name of input blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     * @param info pointer to a pointer to PreProcessInfo structure
     * @param resp Optional: pointer to an already allocated object to contain information in case of failure
     */
    virtual void GetPreProcess(const char* name, const PreProcessInfo** info) const = 0;

    /**
     * @brief Sets new batch size when dynamic batching is enabled in executable network that created this request.
     * @param batch - new batch size to be used by all the following inference calls for this request.
     */
    virtual void SetBatch(int batch) = 0;
};

}  // namespace InferenceEngine
