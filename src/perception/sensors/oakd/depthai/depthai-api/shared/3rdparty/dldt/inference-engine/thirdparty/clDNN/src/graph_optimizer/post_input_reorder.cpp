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

#include "pass_manager.h"
#include "gpu/primitive_gpu_base.h"
#include "fully_connected/fully_connected_params.h"
#include <memory>
#include <stdexcept>

/*
This pass checks if if primitive's input format matches implementation's input format
If not than required reorder is added to the network.
*/

/*
Add a reorder in between node and usr with reorder_layout as layout
*/
program_node& post_input_reorder::add_reorder(program_impl& p,
                                              program_node* node,
                                              program_node* usr,
                                              const layout& reorder_layout) {
    auto new_reorder = std::make_shared<reorder>(node->id() + "_reorder_" + usr->id(), node->id(), reorder_layout);
    auto& new_reorder_node = p.get_or_create(new_reorder);

    // ToDo: add a method to program_impl class which adds an intermediate node given a node and its user
    auto it = std::find(usr->get_dependencies().begin(), usr->get_dependencies().end(), node);
    if (it == usr->get_dependencies().end()) {
        throw std::runtime_error("Inconcistency in topology description: user of a node is not present among its dependecies.");
    }
    auto idx = it - usr->get_dependencies().begin();
    if (idx < 0 || (size_t)idx >= usr->get_dependencies().size()) {
        throw std::runtime_error("Internal Error: container index out of range exception.");
    }
    p.add_intermediate(new_reorder_node, *usr, idx);
    return new_reorder_node;
}

void post_input_reorder::run(program_impl& p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = *node_itr++;
        const auto impl = node->get_selected_impl();
        // add a reorder if primitive's input format doesn't match implementation's input format
        if (node->is_type<fully_connected>()) {
            const auto& fc_impl = dynamic_cast<gpu::typed_primitive_gpu_impl<fully_connected>&>(*impl);
            const auto& fc_params =
                *static_cast<kernel_selector::fully_connected_params*>(fc_impl._kernel_data.params.get());

            auto layout_format = from_data_layout(fc_params.inputs[0].GetLayout());
            auto& input = node->get_dependencies()[0];
            auto input_layout = input->get_output_layout();

            if (input_layout.format != layout_format) {
                auto previous_layout = node->get_output_layout();
                layout current_layout(input_layout.data_type,
                                      layout_format,
                                      input_layout.size,
                                      input_layout.data_padding);
                auto& reorder = add_reorder(p, input, node, current_layout);
                reorder.get_output_layout(false);
                node->set_output_layout(previous_layout, false);
                reorder.set_selected_impl(reorder.type()->choose_impl(p.get_engine(), reorder));
            }
        }
    }
}
