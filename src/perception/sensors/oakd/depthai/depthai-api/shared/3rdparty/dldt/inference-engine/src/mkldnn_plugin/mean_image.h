// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include "mkldnn_dims.h"
#include "ie_parallel.hpp"
#include <vector>
#include <limits>

namespace MKLDNNPlugin {

class MeanImage {
public:
    MeanImage();

public:
    void Load(const MKLDNNDims& inputDims, InferenceEngine::InputInfo::Ptr inputInfo);
    void Subtract(const MKLDNNDims &inputDims, float *input, InferenceEngine::Layout layout);

    template<typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
    void Subtract(const MKLDNNDims &inputDims, T *input, InferenceEngine::Layout layout) {
        IE_ASSERT(input != nullptr);

        if (inputDims.ndims() != 4) {
            THROW_IE_EXCEPTION << "Expecting input as 4 dimension blob with format NxCxHxW.";
        }

        if (layout != InferenceEngine::NCHW && layout != InferenceEngine::NHWC) {
            THROW_IE_EXCEPTION << "Expecting input layout NCHW or NHWC.";
        }

        int MB = inputDims[0];
        int srcSize = inputDims.size() / MB;

        if (meanBuffer && meanBuffer->size()) {
            const float * meanBufferValues = meanBuffer->readOnly();

            InferenceEngine::parallel_for2d(MB, srcSize, [&](int mb, int i) {
                int buf = input[srcSize * mb + i];
                buf -= meanBufferValues[i];
                if (buf < std::numeric_limits<T>::min()) buf = std::numeric_limits<T>::min();
                if (buf > std::numeric_limits<T>::max()) buf = std::numeric_limits<T>::max();
                input[srcSize * mb + i] = buf;
            });
        } else if (!meanValues.empty()) {
            int C = inputDims[1];
            srcSize /= inputDims[1];

            if (layout == InferenceEngine::NCHW) {
                InferenceEngine::parallel_for3d(MB, C, srcSize, [&](int mb, int c, int i) {
                    int buf = input[srcSize * mb * C + c * srcSize + i];
                    buf -= meanValues[c];
                    if (buf < std::numeric_limits<T>::min()) buf = std::numeric_limits<T>::min();
                    if (buf > std::numeric_limits<T>::max()) buf = std::numeric_limits<T>::max();
                    input[srcSize * mb * C + c * srcSize + i] = buf;
                });
            } else if (layout == InferenceEngine::NHWC) {
                InferenceEngine::parallel_for2d(MB, srcSize, [&](int mb, int i) {
                    for (int c = 0; c < C; c++) {
                        int buf = input[mb * srcSize * C + i * C + c];
                        buf -= meanValues[c];
                        if (buf < std::numeric_limits<T>::min()) buf = std::numeric_limits<T>::min();
                        if (buf > std::numeric_limits<T>::max()) buf = std::numeric_limits<T>::max();
                        input[mb * srcSize * C + i * C + c] = buf;
                    }
                });
            }
        }
    }

private:
    std::vector<float> meanValues;

    InferenceEngine::TBlob<float>::Ptr meanBuffer;
};

}  // namespace MKLDNNPlugin
