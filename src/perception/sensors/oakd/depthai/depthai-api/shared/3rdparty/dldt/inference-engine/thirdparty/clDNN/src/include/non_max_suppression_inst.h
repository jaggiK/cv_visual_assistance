/*
// Copyright (c) 2019 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/non_max_suppression.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<non_max_suppression> : public typed_program_node_base<non_max_suppression> {
    using parent = typed_program_node_base<non_max_suppression>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog)
    {}

    program_node& input_boxes() const {
        return get_dependency(0);
    }

    program_node& input_scores() const {
        return get_dependency(1);
    }

    bool has_num_select_per_class() const { return !get_primitive()->num_select_per_class.empty(); }
    program_node& num_select_per_class_node() const {
        return get_dependency(2);
    }

    bool has_iou_threshold() const { return !get_primitive()->iou_threshold.empty(); }
    program_node& iou_threshold_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        return get_dependency(offset);
    }

    bool has_score_threshold() const { return !get_primitive()->score_threshold.empty(); }
    program_node& score_threshold_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        return get_dependency(offset);
    }
};

using non_max_suppression_node = typed_program_node<non_max_suppression>;

template <>
class typed_primitive_inst<non_max_suppression> : public typed_primitive_inst_base<non_max_suppression> {
    using parent = typed_primitive_inst_base<non_max_suppression>;

public:
    typed_primitive_inst(network_impl& network, non_max_suppression_node const& node)
        : parent(network, node)
    {}

    static layout calc_output_layout(non_max_suppression_node const& node);
    static std::string to_string(non_max_suppression_node const& node);

    memory_impl& input_boxes_mem() const {
        return dep_memory(0);
    }

    memory_impl& input_scores_mem() const {
        return dep_memory(1);
    }

    bool has_num_select_per_class() const { return node.has_num_select_per_class(); }
    memory_impl& num_select_per_class_mem() const {
        return dep_memory(2);
    }

    bool has_iou_threshold() const { return node.has_iou_threshold(); }
    memory_impl& iou_threshold_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        return dep_memory(offset);
    }

    bool has_score_threshold() const { return node.has_score_threshold(); }
    memory_impl& score_threshold_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        return dep_memory(offset);
    }
};

using non_max_suppression_inst = typed_primitive_inst<non_max_suppression>;

}  // namespace cldnn
