// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_mul_add_to_scaleshift_or_power.hpp"

#include <memory>
#include <ngraph_ops/power.hpp>
#include <ngraph_ops/scaleshift.hpp>
#include <vector>
#include <algorithm>

#include "ngraph/graph_util.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "utils/utils.hpp"

CONVERSION_RESULT check_constant(const std::shared_ptr<ngraph::op::Constant>& constant,
                                 const ngraph::Shape& shape) {
    if (!constant) return CONVERSION_RESULT::NONE;

    auto const_shape = constant->get_shape();
    auto input_shape = shape;

    // In case of scalar we will convert it to Power
    if (const_shape.empty() || (const_shape.size() == 1 && const_shape[0] == 1)) {
        return CONVERSION_RESULT::POWER;
    }

    // Align shapes
    size_t max_shape_len = std::max(input_shape.size(), const_shape.size());
    while (const_shape.size() < max_shape_len) const_shape.insert(const_shape.begin(), 1);
    while (input_shape.size() < max_shape_len) input_shape.insert(input_shape.begin(), 1);

    // This is feature dimension index from right side (ex. for NCDHW it's equal to 3).
    const size_t feature_index = input_shape.size() - 2;
    if (const_shape.size() < feature_index) return CONVERSION_RESULT::NONE;

    bool is_power = false;
    auto in_it = const_shape.rbegin();
    auto out_it = input_shape.rbegin();
    for (int idx = 0; in_it != const_shape.rend() && out_it != input_shape.rend(); ++in_it, ++out_it, ++idx) {
        if (idx != feature_index && *in_it != 1) {
            return CONVERSION_RESULT::NONE;
        }

        if (idx == feature_index && *in_it == 1) {
            is_power = true;
        } else if (idx == feature_index && *in_it != *out_it) {
            return CONVERSION_RESULT::NONE;
        }
    }

    return is_power ? CONVERSION_RESULT::POWER : CONVERSION_RESULT::SCALE_SHIFT;
}

void ngraph::pass::ConvertMulAddToScaleShiftOrPower::convert_mul_add_to_scaleshift_or_power() {
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, Shape {1});

    auto weights = std::make_shared<ngraph::op::Constant>(element::f32, Shape {1}, std::vector<float> {0});
    auto bias = std::make_shared<ngraph::op::Constant>(element::f32, Shape {1}, std::vector<float> {0});

    auto mul = std::make_shared<ngraph::op::v1::Multiply>(data_batch, weights);
    auto add = std::make_shared<ngraph::op::v1::Add>(mul, bias);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto add_node = ngraph::as_type_ptr<ngraph::op::v1::Add>(m.get_match_root());

        if (!add_node) {
            return false;
        }

        if (!add_node->get_element_type().is_real()) {
            return false;
        }

        auto add_input_0 = add_node->input(0).get_source_output().get_node_shared_ptr();
        auto add_input_1 = add_node->input(1).get_source_output().get_node_shared_ptr();

        auto mul_node = ngraph::as_type_ptr<ngraph::op::v1::Multiply>(add_input_0);
        auto const_bias_node = ngraph::as_type_ptr<ngraph::op::Constant>(add_input_1);
        if (!mul_node) {
            mul_node = ngraph::as_type_ptr<ngraph::op::v1::Multiply>(add_input_1);
            const_bias_node = ngraph::as_type_ptr<ngraph::op::Constant>(add_input_0);
        }

        auto mul_input_0 = mul_node->input(0).get_source_output().get_node_shared_ptr();
        auto mul_input_1 = mul_node->input(1).get_source_output().get_node_shared_ptr();

        auto data_node = mul_node->input(0).get_source_output();
        auto const_weights_node = ngraph::as_type_ptr<ngraph::op::Constant>(mul_input_1);
        if (!const_weights_node) {
            data_node = mul_node->input(1).get_source_output();
            const_weights_node = ngraph::as_type_ptr<ngraph::op::Constant>(mul_input_0);
        }

        // Check that eltwise is not useless otherwise we remove it
        if (ngraph::op::util::constantIsEqualTo(const_weights_node, 1) &&
            ngraph::op::util::constantIsEqualTo(const_bias_node, 0)) {
            bool has_result_output = false;
            for (const auto & output : add_node->output(0).get_target_inputs()) {
                if (dynamic_cast<ngraph::op::Result*>(output.get_node())) {
                    has_result_output = true;
                }
            }

            auto parent = data_node.get_node_shared_ptr();
            size_t consumers_count = 0;
            for (const auto &output : parent->outputs()) {
                consumers_count += output.get_target_inputs().size();
            }

            if (!has_result_output || consumers_count == 1) {
                if (!std::dynamic_pointer_cast<ngraph::op::Parameter>(parent)) {
                    parent->set_friendly_name(add_node->get_friendly_name());
                }
                // TODO: due to ngraph::replace_node function limitations we have to reconnect output port consumers to the new input
                // using replace_source_output method
                for (auto &input : add_node->output(0).get_target_inputs()) {
                    input.replace_source_output(data_node);
                }
                return true;
            }
        }

        auto res1 = check_constant(const_weights_node, data_node.get_shape());
        auto res2 = check_constant(const_bias_node, mul_node->get_output_shape(0));

        if (res1 == CONVERSION_RESULT::NONE || res2 == CONVERSION_RESULT::NONE ||
            ((res1 == CONVERSION_RESULT::SCALE_SHIFT || res2 == CONVERSION_RESULT::SCALE_SHIFT) && add_node->get_shape().size() < 4)) {
            return false;
        }

        // TODO: in case if scale and shift constants has equal values the best way is to convert them to Power
        if (res1 == CONVERSION_RESULT::SCALE_SHIFT || res2 == CONVERSION_RESULT::SCALE_SHIFT) {
            auto weights_in = ngraph::op::util::normalize_constant(const_weights_node, add_node->get_shape());
            auto biases_in = ngraph::op::util::normalize_constant(const_bias_node, add_node->get_shape());
            if (res1 == CONVERSION_RESULT::POWER)
                weights_in = ngraph::op::util::broadcastTo(weights_in, biases_in->get_shape());
            if (res2 == CONVERSION_RESULT::POWER)
                biases_in = ngraph::op::util::broadcastTo(biases_in, weights_in->get_shape());

            auto scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data_node, weights_in, biases_in);
            scaleshift->set_friendly_name(add_node->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<Node>(scaleshift));
        } else {
            float scale = 0.f, shift = 0.f;
            if (!op::util::get_single_value(const_weights_node, scale)) {
                return false;
            }
            if (!op::util::get_single_value(const_bias_node, shift)) {
                return false;
            }

            auto power = std::make_shared<ngraph::op::PowerIE>(data_node, 1., scale, shift);
            power->set_friendly_name(add_node->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), power);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "CPUFusion.MulAddToScaleShiftOrPower");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
