// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"

#include <string>

namespace MKLDNNPlugin {

/**
 * Utility class to dump blob contant in plain format.
 * Every layout information will be lost.
 *
 * In case of low precision blob it allow to store
 * with using scaling factors per channel.
 * NB! Channel is a second dimension for all blob types.
 */
class BlobDumper {
    InferenceEngine::Blob::Ptr _blob;
    InferenceEngine::Blob::Ptr _scales;

public:
    BlobDumper() = default;
    BlobDumper(const BlobDumper&) = default;
    BlobDumper& operator = (BlobDumper&&) = default;

    explicit BlobDumper(const InferenceEngine::Blob::Ptr blob):_blob(blob) {}

    static BlobDumper read(const std::string &file_path);
    static BlobDumper read(std::istream &stream);

    void dump(const std::string &file_path);
    void dump(std::ostream &stream);

    void dumpAsTxt(const std::string file_path);
    void dumpAsTxt(std::ostream &stream);

    BlobDumper& withScales(InferenceEngine::Blob::Ptr scales);
    BlobDumper& withoutScales();

    const InferenceEngine::Blob::Ptr& getScales() const;

    InferenceEngine::Blob::Ptr get();
    InferenceEngine::Blob::Ptr getRealValue();
};

}  // namespace MKLDNNPlugin
