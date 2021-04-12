/*
// Copyright (c) 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "proposal_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <cmath>
#include <string>
#include <vector>

namespace cldnn {

static void generate_anchors(unsigned base_size,
                             const std::vector<float>& ratios,
                             const std::vector<float>& scales,
                             std::vector<proposal_inst::anchor>& anchors,
                             float coordinates_offset,
                             bool shift_anchors,
                             bool round_ratios);

primitive_type_id proposal::type_id() {
    static primitive_type_base<proposal> instance;
    return &instance;
}

layout proposal_inst::calc_output_layout(proposal_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for proposal_node!");
    auto desc = node.get_primitive();
    layout input_layout = node.get_dependency(cls_scores_index).get_output_layout();

    return layout(input_layout.data_type,
                  format::bfyx,
                  {input_layout.size.batch[0] * desc->post_nms_topn, CLDNN_ROI_VECTOR_SIZE, 1, 1});
}

static inline std::string stringify_vector(std::vector<float> v) {
    std::stringstream s;

    s << "{ ";

    for (size_t i = 0; i < v.size(); ++i) {
        s << v.at(i);
        if (i + 1 < v.size())
            s << ", ";
    }

    s << " }";

    return s.str();
}

// TODO: rename to?
static std::string stringify_port(const program_node& p) {
    std::stringstream res;
    auto node_info = p.desc_to_json();
    node_info->dump(res);

    return res.str();
}

std::string proposal_inst::to_string(proposal_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto scales_parm = desc->scales;

    std::stringstream primitive_description;

    auto swap_xy = desc->swap_xy ? "true" : "false";
    auto initial_clip = desc->initial_clip ? "true" : "false";
    auto round_ratios = desc->round_ratios ? "true" : "false";
    auto shift_anchors = desc->shift_anchors ? "true" : "false";
    auto clip_before_nms = desc->clip_before_nms ? "true" : "false";
    auto clip_after_nms = desc->clip_after_nms ? "true" : "false";
    auto for_deformable = desc->clip_after_nms ? "true" : "false";

    json_composite proposal_info;
    proposal_info.add("cls score", stringify_port(node.cls_score()));
    proposal_info.add("box pred", stringify_port(node.bbox_pred()));
    proposal_info.add("image info", stringify_port(node.image_info()));

    json_composite params;
    params.add("max proposals", desc->max_proposals);
    params.add("iou threshold", desc->iou_threshold);
    params.add("base bbox size", desc->base_bbox_size);
    params.add("min bbox size", desc->min_bbox_size);
    params.add("pre nms topn", desc->pre_nms_topn);
    params.add("post nms topn", desc->post_nms_topn);
    params.add("ratios", stringify_vector(desc->ratios));
    params.add("ratios", stringify_vector(desc->ratios));
    params.add("coordinates offset", desc->coordinates_offset);
    params.add("box coordinate scale", desc->box_coordinate_scale);
    params.add("box size scale", desc->box_size_scale);
    params.add("swap xy", swap_xy);
    params.add("initial clip", initial_clip);
    params.add("round ratios", round_ratios);
    params.add("shift anchors", shift_anchors);
    params.add("clip_before_nms", clip_before_nms);
    params.add("clip_after_nms", clip_after_nms);
    params.add("for_deformable", for_deformable);
    proposal_info.add("params", params);

    node_info->add("proposal info", proposal_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

proposal_inst::typed_primitive_inst(network_impl& network, proposal_node const& node) : parent(network, node) {
    generate_anchors(argument.base_bbox_size,
                     argument.ratios,
                     argument.scales,
                     _anchors,
                     argument.coordinates_offset,
                     argument.shift_anchors,
                     argument.round_ratios);
}

static void generate_anchors(unsigned int base_size,
                             const std::vector<float>& ratios,
                             const std::vector<float>& scales,             // input
                             std::vector<proposal_inst::anchor>& anchors,  // output
                             float coordinates_offset,
                             bool shift_anchors,
                             bool round_ratios) {
    const float base_area = static_cast<float>(base_size * base_size);
    const float half_base_size = base_size * 0.5f;
    const float center = 0.5f * (base_size - coordinates_offset);

    anchors.clear();
    // enumerate all transformed boxes
    for (size_t ratio = 0; ratio < ratios.size(); ++ratio) {
        // transformed width & height for given ratio factors
        float ratio_w;
        float ratio_h;
        if (round_ratios) {
            ratio_w = std::roundf(std::sqrt(base_area / ratios[ratio]));
            ratio_h = std::roundf(ratio_w * ratios[ratio]);
        } else {
            ratio_w = std::sqrt(base_area / ratios[ratio]);
            ratio_h = ratio_w * ratios[ratio];
        }

        for (size_t scale = 0; scale < scales.size(); ++scale) {
            proposal_inst::anchor anchor;
            // transformed width & height for given scale factors
            const float scale_w = 0.5f * (ratio_w * scales[scale] - coordinates_offset);
            const float scale_h = 0.5f * (ratio_h * scales[scale] - coordinates_offset);

            // (x1, y1, x2, y2) for transformed box
            anchor.start_x = center - scale_w;
            anchor.start_y = center - scale_h;
            anchor.end_x = center + scale_w;
            anchor.end_y = center + scale_h;

            if (shift_anchors) {
                anchor.start_x -= half_base_size;
                anchor.start_y -= half_base_size;
                anchor.end_x -= half_base_size;
                anchor.end_y -= half_base_size;
            }

            anchors.push_back(anchor);
        }
    }
}
}  // namespace cldnn
