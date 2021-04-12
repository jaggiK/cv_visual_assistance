// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_ie.hpp"

#include <ie_blob.h>

#include <algorithm>
#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "blob_factory.hpp"
#include "ie_ngraph_utils.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

constexpr ::ngraph::NodeTypeInfo ngraph::op::GenericIE::type_info;
std::vector<InferenceEngine::IShapeInferExtensionPtr> ngraph::op::GenericIE::extensions;

void ngraph::op::GenericIE::addExtension(const InferenceEngine::IShapeInferExtensionPtr& ext) {
    extensions.emplace_back(ext);
}

const std::vector<InferenceEngine::IShapeInferExtensionPtr>& ngraph::op::GenericIE::getExtensions() {
    return extensions;
}

ngraph::op::GenericIE::GenericIE(const ngraph::NodeVector& inputs,
                                 const std::map<std::string, InferenceEngine::Parameter>& params,
                                 const std::string type, const std::vector<PortIE>& outputs)
    : GenericIE(as_output_vector(inputs), params, type, outputs) {}

ngraph::op::GenericIE::GenericIE(const ngraph::OutputVector& inputs,
                                 const std::map<std::string, InferenceEngine::Parameter>& params,
                                 const std::string type, const std::vector<PortIE>& outputs)
    : Op(inputs), params(params), type(type), outputs(outputs), initialized(0) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ngraph::op::GenericIE::copy_with_new_args(const ngraph::NodeVector& new_args) const {
    auto genNode = std::make_shared<GenericIE>(new_args, params, type, outputs);
    genNode->reshape = reshape;
    return genNode;
}

void ngraph::op::GenericIE::validate_and_infer_types() {
    // Try to find extension with shape inference inplementation and apply it
    for (const auto& ext : extensions) {
        InferenceEngine::IShapeInferImpl::Ptr impl;
        InferenceEngine::StatusCode ret = ext->getShapeInferImpl(impl, type.c_str(), nullptr);
        if (ret != InferenceEngine::StatusCode::OK || !impl) continue;

        std::vector<InferenceEngine::Blob::CPtr> inputs;
        std::map<std::string, std::string> parameters;
        std::map<std::string, InferenceEngine::Blob::Ptr> blobs;
        std::vector<InferenceEngine::SizeVector> outShapes;

        for (uint64_t i = 0; i < get_input_size(); i++) {
            PartialShape this_input_shape = get_input_partial_shape(i);

            if (!this_input_shape.is_static())
                THROW_IE_EXCEPTION << "Generic node for layer " << get_friendly_name() << " with type " << type
                                   << " has dynamic input shapes!";

            Shape this_ishape = get_input_shape(i);
            InferenceEngine::SizeVector dims = this_ishape;
            InferenceEngine::Blob::Ptr input = make_blob_with_precision(InferenceEngine::TensorDesc(
                InferenceEngine::details::ngraph::convertPrecision(get_input_element_type(i)), dims,
                InferenceEngine::TensorDesc::getLayoutByDims(dims)));
            inputs.emplace_back(input);
        }

        for (const auto& attr : params) {
            if (attr.second.is<std::string>()) {
                parameters[attr.first] = attr.second.as<std::string>();
            } else if (attr.second.is<InferenceEngine::Blob::CPtr>()) {
                auto cBlob = attr.second.as<InferenceEngine::Blob::CPtr>();
                auto wBlob = std::const_pointer_cast<InferenceEngine::Blob>(cBlob);
                blobs[attr.first] = wBlob;
            } else if (attr.second.is<InferenceEngine::Blob::Ptr>()) {
                auto wBlob = attr.second.as<InferenceEngine::Blob::Ptr>();
                blobs[attr.first] = wBlob;
            } else {
                THROW_IE_EXCEPTION << "Generic node for layer " << get_friendly_name() << " with type " << type
                                   << " has incorrect parameter " << attr.first << "!";
            }
        }

        ret = impl->inferShapes(inputs, parameters, blobs, outShapes, nullptr);

        if (ret != InferenceEngine::StatusCode::OK || outShapes.size() != outputs.size()) continue;

        for (size_t i = 0; i < outputs.size(); i++) {
            const auto& port = outputs[i];
            ngraph::Shape outShape(outShapes[i]);
            auto type = InferenceEngine::details::ngraph::convertPrecision(port.precision);
            set_output_type(i, type, PartialShape(outShape));
        }

        return;
    }

    // Extensions are not loaded when we create nGraph function
    // First call: create node
    if (initialized < 1) {
        if (outputs.size())
            set_output_size(outputs.size());
        for (size_t i = 0; i < outputs.size(); i++) {
            const auto& port = outputs[i];
            ngraph::Shape outShape(port.dims);
            auto type = InferenceEngine::details::ngraph::convertPrecision(port.precision);
            set_output_type(i, type, PartialShape(outShape));
        }
        initialized++;
    } else if (reshape) {
        THROW_IE_EXCEPTION << "IShapeInferExtension wasn't registrated for node " << get_friendly_name()
                           << " with type " << type;
    }
}
