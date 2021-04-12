// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <cmath>
#include "defs.h"
#include "ie_parallel.hpp"

struct jit_uni_softmax_kernel;

static inline
void softmax_many_batches(const float *src_data, float *dst_data, int B, int C, int H, int W) {
    InferenceEngine::parallel_for(B * H * W, [&](size_t i) {
        const float *psrc = src_data + (i / (H * W)) * C * H * W - (i / (H * W)) * H * W;
        float *pdst = dst_data + (i / (H * W)) * C * H * W - (i / (H * W)) * H * W;

        float max = psrc[i];
        for (int c = 0; c < C; c++) {
            float val = psrc[c * H * W + i];
            if (val > max) max = val;
        }

        float expSum = 0;
        for (int c = 0; c < C; c++) {
            pdst[c * H * W + i] = exp(psrc[c * H * W + i] - max);
            expSum += pdst[c * H * W + i];
        }

        for (int c = 0; c < C; c++) {
            pdst[c * H * W + i] = pdst[c * H * W + i] / expSum;
        }
    });
}

class SoftmaxGeneric {
public:
    SoftmaxGeneric();

    void execute(const float *src_data, float *dst_data, int B, int C, int H, int W);

private:
    int block_size;
    std::shared_ptr<jit_uni_softmax_kernel> softmax_kernel;
};

