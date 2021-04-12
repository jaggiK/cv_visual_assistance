// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "ie_parallel.hpp"
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SqueezeImpl: public ExtLayerBase {
public:
    explicit SqueezeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 1 && layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            SizeVector data_dims = layer->insData[0].lock()->getTensorDesc().getDims();
            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (data_dims.size() < dst_dims.size())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";

            if (layer->insData.size() == 1)
                addConfig(layer, { { ConfLayout::PLN, false, 0 } }, { { ConfLayout::PLN, false, 0 } });
            else
                addConfig(layer, { { ConfLayout::PLN, false, 0 }, { ConfLayout::PLN, false, 0 } }, { { ConfLayout::PLN, false, 0 } });

            // WA to enable the implementation only for equal input and output precisions
            confs[0].inConfs[0].desc.setPrecision(confs[0].outConfs[0].desc.getPrecision());
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const uint8_t *src = inputs[0]->cbuffer().as<uint8_t *>() + inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding()*inputs[0]->element_size();
        uint8_t* dst = outputs[0]->cbuffer().as<uint8_t *>() + outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding()*outputs[0]->element_size();

        if (src != dst) {
            size_t srcSize = inputs[0]->byteSize();
            size_t dstSize = outputs[0]->byteSize();
            simple_copy(dst, dstSize, src, srcSize);
        }

        return OK;
    }
};

REG_FACTORY_FOR(ImplFactory<SqueezeImpl>, Squeeze);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
