// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>
#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/interp.hpp>

#include "ngraph/op/experimental/layers/interpolate.hpp"
namespace ngraph {
namespace pass {

class ConvertInterpolateToInterpOrResample;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertInterpolateToInterpOrResample: public ngraph::pass::GraphRewrite {
public:
    ConvertInterpolateToInterpOrResample() : GraphRewrite() {
        convert_interpolate_to_interp_or_resample();
    }

private:
    void convert_interpolate_to_interp_or_resample() {
        auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto interpolate = std::make_shared<ngraph::op::Interpolate>(data, shp, ngraph::op::InterpolateAttrs());

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto interpolate = std::dynamic_pointer_cast<ngraph::op::Interpolate> (m.get_match_root());

            auto data_node = interpolate->input_value(0);
            auto out_shape_node = std::dynamic_pointer_cast<ngraph::op::Constant>(interpolate->input_value(1).get_node_shared_ptr());
            auto interpolate_attrs = interpolate->get_attrs();
            auto input_shape = data_node.get_shape();

            if (!out_shape_node) {
                return false;
            }

            auto out_spatial_shape = out_shape_node->get_vector<int64_t> ();
            if (out_spatial_shape.size() != 2 && out_spatial_shape.size() != 3) {
                return false;
            }
            // Interpolate can be converted when interpolation is performed over spatial dimensions only
            if (interpolate_attrs.axes != AxisSet{2, 3} && interpolate_attrs.axes != AxisSet{2, 3, 4}) {
                return false;
            }

            if (interpolate_attrs.axes.size() == 2 && std::set<std::string>{"nearest", "cubic", "area"}.count(interpolate_attrs.mode) == 0) {
                auto attrs = ngraph::op::InterpolateIEAttrs();
                attrs.pad_beg = interpolate_attrs.pads_begin[0];
                attrs.pad_end = interpolate_attrs.pads_end[0];
                attrs.height = out_spatial_shape[0];
                attrs.width = out_spatial_shape[1];
                attrs.align_corners = interpolate_attrs.align_corners;
                attrs.mode = interpolate_attrs.mode;
                attrs.antialias = interpolate_attrs.antialias;

                auto interp = std::make_shared<ngraph::op::Interp>(data_node, attrs);
                interp->set_friendly_name(m.get_match_root()->get_friendly_name());
                ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<ngraph::Node>(interp));
            } else if (interpolate_attrs.pads_begin[0] == 0 && interpolate_attrs.pads_end[0] == 0 && !interpolate_attrs.align_corners) {
                auto attrs = ngraph::op::ResampleIEAttrs();
                attrs.mode = interpolate_attrs.mode;
                attrs.antialias = interpolate_attrs.antialias;

                std::shared_ptr<Node> resample;

                // In case if output shape differ only in spatial dims and can be produced by using factor we set factor attr
                bool has_same_factor(true);
                int64_t factor(0);
                for (size_t i = 0; i < out_spatial_shape.size(); ++i) {
                    if (out_spatial_shape[i] % input_shape[i + 2] == 0) {
                        int64_t f = out_spatial_shape[i] / input_shape[i + 2];
                        if (factor == 0) {
                            factor = f;
                        } else if (factor != f) {
                            has_same_factor = false;
                        }
                    } else {
                        has_same_factor = false;
                    }
                }

                if (has_same_factor && factor != 0) {
                    attrs.factor = factor;
                    resample = std::make_shared<ngraph::op::ResampleV2>(data_node, attrs);
                } else {
                    // first concatenates [N,C] shapes from the input tensor with the Interpolate second input value to
                    // create the desired output shape for the Resample
                    auto output_shape = out_spatial_shape;
                    output_shape.insert(output_shape.begin(), input_shape[0]);
                    output_shape.insert(output_shape.begin() + 1, input_shape[1]);
                    auto constant = std::make_shared<ngraph::op::Constant>(out_shape_node->get_element_type(), Shape{output_shape.size()}, output_shape);
                    resample = std::make_shared<ngraph::op::ResampleV2>(data_node, constant, attrs);
                }

                resample->set_friendly_name(m.get_match_root()->get_friendly_name());
                ngraph::replace_node(m.get_match_root(), resample);
            } else {
                return false;
            }
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate, "ConvertInterpolateToInterpOrResample");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
