//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "prior_box.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PriorBox::type_info;

op::PriorBox::PriorBox(const Output<Node>& layer_shape,
                       const Output<Node>& image_shape,
                       const PriorBoxAttrs& attrs)
    : Op({layer_shape, image_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::PriorBox::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    auto layer_shape_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          layer_shape_et.compatible(element::Type_t::i64),
                          "layer shape input must have element type i64, but has ",
                          layer_shape_et);

    auto image_shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          image_shape_et.compatible(element::Type_t::i64),
                          "image shape input must have element type i64, but has ",
                          image_shape_et);

    auto layer_shape_rank = get_input_partial_shape(0).rank();
    auto image_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          layer_shape_rank.compatible(image_shape_rank),
                          "layer shape input rank ",
                          layer_shape_rank,
                          " must match image shape input rank ",
                          image_shape_rank);

    set_input_is_relevant_to_shape(0);

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(0).get_node_shared_ptr()))
    {
        NODE_VALIDATION_CHECK(this,
                              shape_size(const_shape->get_shape()) == 2,
                              "Layer shape must have rank 2",
                              const_shape->get_shape());

        auto layer_shape = const_shape->get_shape_val();

        set_output_type(0,
                        element::f32,
                        Shape{2, 4 * layer_shape[0] * layer_shape[1] * number_of_priors(m_attrs)});
    }
    else
    {
        set_output_type(0, element::f32, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::PriorBox::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<PriorBox>(new_args.at(0), new_args.at(1), m_attrs);
}

size_t op::PriorBox::number_of_priors(const PriorBoxAttrs& attrs)
{
    // Starting with 0 number of prior and then various conditions on attributes will contribute
    // real number of prior boxes as PriorBox is a fat thing with several modes of
    // operation that will be checked in order in the next statements.
    size_t num_priors = 0;

    // Total number of boxes around each point; depends on whether flipped boxes are included
    // plus one box 1x1.
    size_t total_aspect_ratios = normalized_aspect_ratio(attrs.aspect_ratio, attrs.flip).size();

    if (attrs.scale_all_sizes)
        num_priors = total_aspect_ratios * attrs.min_size.size() + attrs.max_size.size();
    else
        num_priors = total_aspect_ratios + attrs.min_size.size() - 1;

    if (!attrs.fixed_size.empty())
        num_priors = total_aspect_ratios * attrs.fixed_size.size();

    for (auto density : attrs.density)
    {
        auto rounded_density = static_cast<size_t>(density);
        auto density_2d = (rounded_density * rounded_density - 1);
        if (!attrs.fixed_ratio.empty())
            num_priors += attrs.fixed_ratio.size() * density_2d;
        else
            num_priors += total_aspect_ratios * density_2d;
    }

    return num_priors;
}

std::vector<float> op::PriorBox::normalized_aspect_ratio(const std::vector<float>& aspect_ratio,
                                                         bool flip)
{
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio)
    {
        unique_ratios.insert(std::round(ratio * 1e6) / 1e6);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6) / 1e6);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}