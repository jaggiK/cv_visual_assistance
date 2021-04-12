// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_ie.hpp"

#include <memory>
#include <string>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::NormalizeIE::type_info;

op::NormalizeIE::NormalizeIE(const Output<Node>& data, const Output<Node>& weights, float eps, bool across_spatial,
                             bool channel_shared)
    : Op({data, weights}), m_eps(eps), m_across_spatial(across_spatial), m_channel_shared(channel_shared) {
    constructor_validate_and_infer_types();
}

void op::NormalizeIE::validate_and_infer_types() {
    element::Type arg_type = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);

    const PartialShape& input_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_shape.rank().is_dynamic() || static_cast<size_t>(input_shape.rank()) >= 2 &&
                                                                 static_cast<size_t>(input_shape.rank()) <= 4,
                          "Argument must have rank >= 2 and <= 4 (argument shape: ", input_shape, ").");
}

shared_ptr<Node> op::NormalizeIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::NormalizeIE>(new_args.at(0), new_args.at(1), m_eps, m_across_spatial, m_channel_shared);
}
