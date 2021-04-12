// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/gather_ie.hpp>

#include "ngraph/op/gather.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"


namespace ngraph {
namespace pass {

class ConvertGatherToGatherIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGatherToGatherIE : public ngraph::pass::GraphRewrite {
public:
    ConvertGatherToGatherIE() : GraphRewrite() {
        convert_gather_to_gather_ie();
    }

private:
    void convert_gather_to_gather_ie() {
        auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
        auto input_1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
        auto input_2 = std::make_shared<pattern::op::Label>(element::i64, Shape{});
        auto gather = std::make_shared<ngraph::op::v1::Gather>(input_0, input_1, input_2);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
            auto gather = std::dynamic_pointer_cast<ngraph::op::v1::Gather>(m.get_match_root());
            if (!gather) {
                return false;
            }

            auto axes_node = gather->input(2).get_source_output().get_node_shared_ptr();
            auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(axes_node);
            if (!axes_constant) {
                return false;
            }
            auto axis = axes_constant->get_vector<int64_t>()[0];

            // if the input with indices is scalar we need to unsqueeze it to 1D so plugins which do not support 0D can
            // execute this layer. Then we need to squeeze the axis dimension to restore original shape of gather output
            auto indices = gather->input(1).get_source_output();
            auto gather_output_shape = gather->output(0).get_shape();
            bool squeeze_gather_output = false;
            if (indices.get_shape().empty()) {
                squeeze_gather_output = true;
                gather_output_shape.insert(gather_output_shape.begin() + axis, 1);
                indices = std::make_shared<ngraph::op::Unsqueeze>(indices.get_node_shared_ptr(),
                                                                  op::Constant::create(element::i64, Shape{1}, {0}));
            }
            auto gather_ie = std::make_shared<ngraph::op::GatherIE>(gather->input(0).get_source_output(),
                                                                    indices,
                                                                    axis,
                                                                    gather_output_shape);
            if (squeeze_gather_output) {
                auto sq = std::make_shared<ngraph::op::Squeeze>(gather_ie,
                                                                op::Constant::create(element::i64, Shape{1}, {axis}));
                sq->set_friendly_name(gather->get_friendly_name());
                ngraph::replace_node(gather, sq);
            } else {
                gather_ie->set_friendly_name(gather->get_friendly_name());
                ngraph::replace_node(gather, gather_ie);
            }
            return true;
        };

        auto m1 = std::make_shared<ngraph::pattern::Matcher>(gather, "ConvertGatherToGatherIE");
        this->add_matcher(m1, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
