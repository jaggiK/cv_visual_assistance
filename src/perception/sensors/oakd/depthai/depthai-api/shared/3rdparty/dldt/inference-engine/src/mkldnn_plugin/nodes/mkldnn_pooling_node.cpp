// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pooling_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include "mkldnn_conv_node.h"
#include "mkldnn_concat_node.h"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPoolingNode::MKLDNNPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket)
        : MKLDNNNode(layer, eng, socket) {}

void MKLDNNPoolingNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    auto * poolingLayer = dynamic_cast<PoolingLayer*>(getCnnLayer().get());
    if (poolingLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert pooling layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    type = poolingLayer->_type;
    exclude_pad = poolingLayer->_exclude_pad;

    inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    outputPrecision = getCnnLayer()->outData[0]->getPrecision();
    // Dirty WA to support stat based quantization approach
    if (this->getCnnLayer()->precision != Precision::I8) {
        if (type == PoolingLayer::MAX) {
            // MKLDNN supports only equal precisions for input and output
            outputPrecision = inputPrecision;
        } else if (type == PoolingLayer::AVG) {
            outputPrecision = Precision::FP32;
        }
    }

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    invertVectorCopyUtoI(poolingLayer->_stride, stride);
    invertVectorCopyUtoI(poolingLayer->_kernel, kernel);
    auto allPads = getPaddings(*poolingLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

    auto parentDims = getParentEdgeAt(0)->getDims();
    auto childDims = getChildEdgeAt(0)->getDims();
    if ((parentDims.ndims() < 4) || (parentDims.ndims() > 5))
        THROW_IE_EXCEPTION << "Pooling layer. Unsupported mode. Only 4D and 5D blobs are supported as input.";

    for (int i = 0; i < paddingR.size(); i++) {
        int krn = kernel[i];
        int src = getParentEdgeAt(0)->getDims()[2 + i];
        int dst = getChildEdgeAt(0)->getDims()[2 + i];

        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }
    if (inputPrecision == Precision::I8 || inputPrecision == Precision::U8) {
        // i8 layers supports only ndhwc and nhwc layouts
        MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, parentDims.ndims() == 5 ? memory::format::ndhwc : memory::format::nhwc};
        MKLDNNMemoryDesc out_candidate{childDims, outputDataType, parentDims.ndims() == 5 ? memory::format::ndhwc : memory::format::nhwc};
        createDescriptor({ in_candidate }, { out_candidate });
    } else if ((parentDims.ndims() == 4 || parentDims.ndims() == 5) && parentDims[1] == 1) {
        inputDataType = memory::f32;
        outputDataType = memory::f32;
        // WA. We should force planar layout since it provides better performance
        MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, parentDims.ndims() == 5 ? memory::format::ncdhw : memory::format::nchw};
        MKLDNNMemoryDesc out_candidate{childDims, outputDataType, parentDims.ndims() == 5 ? memory::format::ncdhw : memory::format::nchw};
        createDescriptor({ in_candidate }, { out_candidate });
    } else {
        inputDataType = memory::f32;
        outputDataType = memory::f32;
        // It doesn't support any format
        for (auto format : getAvailableFormatsForDims(parentDims)) {
            MKLDNNMemoryDesc in_candidate{parentDims, inputDataType, format};
            MKLDNNMemoryDesc out_candidate{childDims, outputDataType, format};
            createDescriptor({in_candidate}, {out_candidate});
        }
    }
}

void MKLDNNPoolingNode::createPrimitive() {
    if (prim)
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr, true);

    auto prim_desc = createPrimitiveDescriptor<pooling_forward::primitive_desc, pooling_forward::desc>(attr);

    prim.reset(new pooling_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                   getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNPoolingNode::created() const {
    return getType() == Pooling;
}

void MKLDNNPoolingNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                         const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);

    algorithm alg;
    if (type == PoolingLayer::PoolType::AVG) {
        bool not_zero_l = false;
        for (auto lr : paddingL) {
            if (lr) {
                not_zero_l = true;
                break;
            }
        }
        if (!exclude_pad && not_zero_l)
            alg = pooling_avg_include_padding;
        else
            alg = pooling_avg_exclude_padding;
    } else if (type == PoolingLayer::PoolType::MAX) {
        alg = pooling_max;
    } else {
        // TODO: Handle rest of the possible: STOCH, ROI, SPACIAL_PYRAMID
        THROW_IE_EXCEPTION << "Unsupported pooling type";
    }

    std::shared_ptr<pooling_forward::desc> desc_ptr(
            new pooling_forward::desc(prop_kind::forward_scoring, alg,
                                      in_candidate, out_candidate,
                                      stride, kernel, paddingL, paddingR,
                                      mkldnn::padding_kind::zero));

    bool not_zero_r = false;
    for (auto pr : paddingR) {
        if (pr) {
            not_zero_r = true;
            break;
        }
    }
    if (alg == pooling_avg_include_padding && not_zero_r) {
        // In case of AVG including paddings the norm coeff should be calculated
        // with tacking into account original pads. So we need to restore
        // original values (R_padding = L_padding).
        //
        // WA. Because mkldnn uses different formula to calculate AVG norm coeff
        //     in compare with Caffe. In mkldnn coeff is always 1/(KH*KW)
        for (int i = 0; i < paddingL.size(); i++) desc_ptr->data.padding[1][i] = paddingL[i];
    }

    descs.emplace_back(desc_ptr);
}

void MKLDNNPoolingNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    mkldnn::primitive_attr attr;
    setPostOps(attr);

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), attr);
        while (itpd.is_not_end()) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getSrcMemDesc(itpd, i));
                config.inConfs.push_back(dataConfig);
            }

            std::vector<mkldnn::memory::format> outFormats;
            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getDstMemDesc(itpd, i));
                config.outConfs.push_back(dataConfig);

                auto primDesc = itpd.fetch();
                auto dstPrimDesc = mkldnn_primitive_desc_query_pd(primDesc.get(), mkldnn::convert_to_c(dst_pd), 0);
                if (dstPrimDesc) {
                    outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));
                } else {
                    // This path is needed to correctly handle Deconvolution node
                    auto diffSrcPrimDesc = mkldnn_primitive_desc_query_pd(primDesc.get(), mkldnn::convert_to_c(diff_src_pd), 0);
                    if (diffSrcPrimDesc) {
                        outFormats.emplace_back(static_cast<memory::format>(itpd.diff_src_primitive_desc().desc().data.format));
                    }
                }
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
            itpd++;
        }
    }
}

void MKLDNNPoolingNode::initDescriptor(const InferenceEngine::LayerConfig &config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    std::vector<InferenceEngine::TensorDesc> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.push_back(inConf.desc);
    std::vector<InferenceEngine::TensorDesc> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.push_back(outConf.desc);
    createDescriptor({inDescs}, {outDescs});

    mkldnn::primitive_attr attr;
    setPostOps(attr);

    InferenceEngine::LayerConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;
    for (size_t j = 0; j < descs.size(); j++) {
        const auto &desc = descs[j];
        std::shared_ptr<primitive_desc_iterator> itpd;

        itpd = std::make_shared<primitive_desc_iterator>(desc.createPrimitiveDescriptorIterator(getEngine(), attr));

        while (itpd->is_not_end()) {
            InferenceEngine::LayerConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(*itpd, i);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(*itpd, i);
                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd->get_impl_info_str().c_str());
            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    THROW_IE_EXCEPTION << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (j == descs.size() - 1) {
                if (impl_type == selectedPD->getImplementationType()) {
                    rightConfig = config;
                }
            }
            selected_count++;
            (*itpd)++;
        }
    }

    if (descs.empty()) {
        const auto& selectedConfig = selectedPD->getConfig();
        if (selectedConfig.inConfs.size() != config.inConfs.size() || selectedConfig.outConfs.size() != config.outConfs.size())
            return;

        for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {
            if (selectedConfig.inConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.inConfs[i].desc, config.inConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (selectedConfig.outConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.outConfs[i].desc, config.outConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }
        rightConfig = config;
    }

    selectedPD->getConfig() = rightConfig;
}

void MKLDNNPoolingNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            if (initWeights) {
                MKLDNNDims weightsDims({static_cast<ptrdiff_t>(rnd_up(getParentEdgeAt(0)->getDims()[1], 16))});
                MKLDNNMemoryDesc weightsDataDesc = {{(uint32_t)weightsDims[0]}, memory::f32, memory::x};

                auto cropLowDataMem = std::make_shared<MKLDNNMemory>(getEngine());
                cropLowDataMem->Create(weightsDataDesc, quantizeNode->getCropLowPtr());

                auto cropHighDataMem = std::make_shared<MKLDNNMemory>(getEngine());
                cropHighDataMem->Create(weightsDataDesc, quantizeNode->getCropHighPtr());

                auto inputScaleDataMem = std::make_shared<MKLDNNMemory>(getEngine());
                inputScaleDataMem->Create(weightsDataDesc, quantizeNode->getInputScalePtr());

                auto inputShiftDataMem = std::make_shared<MKLDNNMemory>(getEngine());
                inputShiftDataMem->Create(weightsDataDesc, quantizeNode->getInputShiftPtr());

                auto outputScaleDataMem = std::make_shared<MKLDNNMemory>(getEngine());
                outputScaleDataMem->Create(weightsDataDesc, quantizeNode->getOutputScalePtr());

                auto outputShiftDataMem = std::make_shared<MKLDNNMemory>(getEngine());
                outputShiftDataMem->Create(weightsDataDesc, quantizeNode->getOutputShiftPtr());

                PostOpsIntBlobMemory.push_back(cropLowDataMem);
                PostOpsIntBlobMemory.push_back(cropHighDataMem);
                PostOpsIntBlobMemory.push_back(inputScaleDataMem);
                PostOpsIntBlobMemory.push_back(inputShiftDataMem);
                PostOpsIntBlobMemory.push_back(outputScaleDataMem);
                PostOpsIntBlobMemory.push_back(outputShiftDataMem);

                ops.append_quantization(quantizeNode->getAlgorithm(), quantizeNode->getCropLowPtr(), quantizeNode->getCropHighPtr(),
                                                                      quantizeNode->getInputScalePtr(), quantizeNode->getInputShiftPtr(),
                                                                      quantizeNode->getOutputScalePtr(), quantizeNode->getOutputShiftPtr());

                blob_idx += 6;
            } else {
                ops.append_quantization(quantizeNode->getAlgorithm(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            }
        }
    }

    attr.set_post_ops(ops);
}

REG_MKLDNN_PRIM_FOR(MKLDNNPoolingNode, Pooling);
