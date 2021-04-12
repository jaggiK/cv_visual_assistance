// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/crop_ie.hpp>

#include "ngraph/op/strided_slice.hpp"

namespace ngraph {
namespace pass {

class ConvertStridedSliceToCrop;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertStridedSliceToCrop: public ngraph::pass::GraphRewrite {
public:
    ConvertStridedSliceToCrop() : GraphRewrite() {
        convert_strided_slice_to_crop();
    }

private:
    void convert_strided_slice_to_crop() {
        auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto m_begin = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto m_end = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto m_stride = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto m_slice = std::make_shared<ngraph::op::v1::StridedSlice>(data, m_begin, m_end, m_stride, begin_mask, end_mask);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto slice = std::dynamic_pointer_cast<ngraph::op::v1::StridedSlice> (m.get_match_root());
            if (!slice) {
                return false;
            }

            auto data_node = slice->get_argument(0);
            auto begin_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->get_argument(1));
            auto end_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->get_argument(2));
            auto stride_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->get_argument(3));

            auto output_shape = slice->get_output_shape(0);

            auto partial_input_shape = slice->get_input_partial_shape(0);

            if (!begin_node || !end_node || !stride_node || partial_input_shape.is_dynamic()) {
                return false;
            }

            auto input_shape = slice->get_input_shape(0);
            // MKLDNN: "Crop supports only 2d, 4d and 5d blobs."
            if (input_shape.size() != 2 && input_shape.size() != 4 && input_shape.size() != 5) {
                return false;
            }

            auto begin = begin_node->get_vector<int64_t>();
            auto end = end_node->get_vector<int64_t>();
            auto strides = stride_node->get_vector<int64_t>();

            bool ones_stride = true;
            for (auto & s : strides) {
                if (s != 1) ones_stride = false;
            }

            if (!ones_stride) return false;

            auto convert_to_set = [](const std::vector<int64_t> mask) {
                AxisSet axis_set{};
                for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
                    if (mask[i] == 1) {
                        axis_set.emplace(i);
                    }
                }
                return axis_set;
            };

            auto shrink_axis_mask = convert_to_set(slice->get_shrink_axis_mask());
            auto new_axis_mask = convert_to_set(slice->get_new_axis_mask());
            auto ellipsis_mask = convert_to_set(slice->get_ellipsis_mask());
            auto begin_mask = convert_to_set(slice->get_begin_mask());
            auto end_mask = convert_to_set(slice->get_end_mask());

            std::vector<int64_t> reshape_pattern,
                                 axes,
                                 offset,
                                 dim;

            size_t input_shape_idx = 0;
            uint64_t uniq_id = 0;
            for (size_t axis = 0; axis < begin.size(); ++axis) {
                // add dimensions hidden under the ellipsis mask if ellipsis mask is set
                if (ellipsis_mask.count(axis)) {
                    // only one bit in ellipsis mask is allowed
                    int num_new_axis_after_ellipses = 0;
                    int num_input_axis_before_ellipses = 0;
                    for (size_t i = 0; i < axis; ++i) {
                        if (!new_axis_mask.count(i))
                            num_input_axis_before_ellipses++;
                    }
                    for (size_t i = axis + 1; i < begin.size(); ++i) {
                        if (new_axis_mask.count(i))
                            num_new_axis_after_ellipses++;
                    }

                    // -1 because it's a position of ellipses
                    unsigned long num_input_axis_after_ellipses = (begin.size() - axis - num_new_axis_after_ellipses - 1);
                    unsigned long num_of_hidden_dims = input_shape.size() - num_input_axis_after_ellipses
                                                       - num_input_axis_before_ellipses;
                    for (size_t i = 0; i < num_of_hidden_dims; ++i) {
                        axes.emplace_back(uniq_id);
                        uniq_id++;
                        reshape_pattern.emplace_back(input_shape[input_shape_idx]);
                        offset.emplace_back(0);

                        dim.emplace_back(input_shape[input_shape_idx]);
                        input_shape_idx++;
                    }
                } else {
                    // add new single dimension if new_axis_mask is set
                    if (new_axis_mask.count(axis)) {
                        reshape_pattern.emplace_back(1);
                        dim.emplace_back(1);
                        offset.emplace_back(0);
                    } else if (shrink_axis_mask.count(axis)) {
                        // skip this dimension if shrink_axis_mask is set (input_shape_idx++)
                        dim.emplace_back(1);
                        offset.emplace_back(0);
                        reshape_pattern.emplace_back(1);
                        input_shape_idx++;
                    } else {
                        // calculate dimension using begin, end, begin_mask, end_mask, stride
                        reshape_pattern.emplace_back(input_shape[input_shape_idx]);

                        int64_t lb = begin[axis];
                        int64_t ub = end[axis];

                        // convert negative indexes to positive
                        if (lb < 0)
                            lb = std::max(static_cast<int64_t>(input_shape[input_shape_idx]) + lb,
                                    static_cast<int64_t>(0));
                        if (ub < 0)
                            ub = std::max(static_cast<int64_t>(input_shape[input_shape_idx]) + ub,
                                    static_cast<int64_t>(0));

                        // apply restrictions when begin or end values more/less than max/min possible values.
                        lb = std::min(static_cast<int64_t>(input_shape[input_shape_idx]), lb);
                        ub = std::min(static_cast<int64_t>(input_shape[input_shape_idx]), ub);

                        offset.emplace_back(lb);

                        // set default value for stride or use given value
                        int64_t stride = 1;
                        if (strides.size() > axis)
                            stride = strides[axis];

                        int64_t dimension = 0;
                        if (stride < 0) {
                            // apply masks
                            if (begin_mask.count(axis))
                                lb = static_cast<int64_t>(input_shape[input_shape_idx]) - 1;
                            if (end_mask.count(axis))
                                ub = -1;

                            lb = std::min(lb, static_cast<int64_t>(input_shape[input_shape_idx]) - 1);
                            lb -= 1;  // we always get 1st element, so we need decrease range
                            if (ub <= lb)
                                dimension = (ub - lb) / stride + 1;
                        } else {
                            // apply masks
                            if (begin_mask.count(axis))
                                lb = 0;
                            if (end_mask.count(axis))
                                ub = static_cast<int64_t>(input_shape[input_shape_idx]);

                            lb += 1;  // we always get 1st element, so we need decrease range
                            if (ub >= lb)
                                dimension = (ub - lb) / stride + 1;
                        }

                        dim.emplace_back(dimension);
                        input_shape_idx++;
                    }
                    axes.emplace_back(uniq_id);
                    uniq_id++;
                }
            }
            for (; input_shape_idx < input_shape.size(); ++input_shape_idx) {
                reshape_pattern.emplace_back(input_shape[input_shape_idx]);
                offset.emplace_back(0);
                dim.emplace_back(input_shape[input_shape_idx]);
                axes.emplace_back(uniq_id);
                uniq_id++;
            }

            // CLDNN: if (cropLayer->axis[i] < 0 || cropLayer->axis[i] > 3) -> invalid crop axis
            if (axes.size() > 4) {
                return false;
            }

            // NODES

            // Reshape in case of new axis
            if (!new_axis_mask.empty()) {
                auto new_shape = std::make_shared<ngraph::op::Constant>(element::i64,
                        ngraph::Shape{reshape_pattern.size()}, reshape_pattern);
                data_node = std::make_shared<ngraph::op::v1::Reshape>(data_node, new_shape, true);
                data_node->set_friendly_name("slice/DynReshape_before");
            }

            // Crop
            data_node = std::make_shared<ngraph::op::CropIE> (data_node, axes, dim, offset);
            data_node->set_friendly_name(slice->get_friendly_name());

            // Reshape in case of deleting of axis
            if (!shrink_axis_mask.empty()) {
                auto new_shape = std::make_shared<ngraph::op::Constant>(element::i64, ngraph::Shape{output_shape.size()},
                        output_shape);
                data_node = std::make_shared<ngraph::op::v1::Reshape>(data_node, new_shape, true);
                data_node->set_friendly_name("slice/DynReshape_after");
            }

            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<ngraph::Node>(data_node));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(m_slice, "ConvertStridedSliceToCrop");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
