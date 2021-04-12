/*
// Copyright (c) 2016-2019 Intel Corporation
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

#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "internal_primitive.h"
#include "internal_primitive_type_base.h"
#include "layout_optimizer.h"
#include "pass_manager.h"
#include "primitive_type.h"
#include "program_dump_graph.h"
#include "program_impl.h"
#include "sliding_window_utils.h"

#include "roi_pooling_inst.h"
#include "reorg_yolo_inst.h"
#include "eltwise_inst.h"
#include "softmax_inst.h"
#include "permute_inst.h"
#include "custom_gpu_primitive_inst.h"
#include "binary_convolution_inst.h"
#include "resample_inst.h"
#include "reshape_inst.h"
#include "activation_inst.h"
#include "scale_inst.h"
#include "convolution_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "deconvolution_inst.h"
#include "detection_output_inst.h"
#include "input_layout_inst.h"
#include "lstm_inst.h"
#include "lstm_elt_inst.h"
#include "lstm_gemm_inst.h"
#include "mutable_data_inst.h"
#include "pooling_inst.h"
#include "primitive_inst.h"
#include "prior_box_inst.h"
#include "proposal_inst.h"
#include "reorder_inst.h"
#include "split_inst.h"
#include "to_string_utils.h"

#include "gpu/ocl_toolkit.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

program::program(engine const& engine, topology const& topology, build_options const& options)
    : _impl(engine.get()->build_program(*topology.get(), options).detach()) {}

void program::retain() {
    _impl->add_ref();
}

void program::release() {
    _impl->release();
}

program_impl::program_impl(engine_impl& engine_ref,
                           topology_impl const& topology,
                           build_options const& options,
                           bool is_internal,
                           bool no_optimizations)
    : engine(&engine_ref),
      options(options),
      processing_order(*new nodes_ordering) {
    set_options();
    pm = std::unique_ptr<pass_manager>(new pass_manager(*this));
    prepare_nodes(topology);
    if (no_optimizations) {
        init_graph();
    } else {
        build_program(is_internal);
    }
}

program_impl::program_impl(engine_impl& engine_ref,
                           std::set<std::shared_ptr<program_node>> const& nodes,
                           build_options const& options,
                           bool is_internal)
    : engine(&engine_ref),
      options(options),
      processing_order(*new nodes_ordering) {
    set_options();
    pm = std::unique_ptr<pass_manager>(new pass_manager(*this));
    prepare_nodes(nodes);
    build_program(is_internal);
}

program_impl::~program_impl() = default;

program_node& program_impl::get_node(primitive_id const& id) {
    try {
        return *nodes_map.at(id);
    } catch (...) {
        throw std::runtime_error("Program doesn't contain primtive node: " + id);
    }
}

program_node const& program_impl::get_node(primitive_id const& id) const {
    try {
        return *nodes_map.at(id);
    } catch (...) {
        throw std::runtime_error("Program doesn't contain primtive node: " + id);
    }
}

// TODO: Remove once we will get full support for input/output padding in all primitive implementations.
bool program_impl::analyze_output_size_handling_need() {
    bool handling_needed = false;

    // Calculate output size and compare with specified.
    for (const auto& node : processing_order) {
        if (node->is_type<convolution>()) {
            auto& prim_node = node->as<convolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range(
                {0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1], prim->output_size.spatial[2]},
                1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range =
                calc_sliding_window_output_range<swor_mode::all>(prim_node.input().get_output_layout().size,
                                                                 filter_size,
                                                                 prim->input_offset,
                                                                 prim->stride,
                                                                 prim->dilation,
                                                                 true,
                                                                 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        } else if (node->is_type<binary_convolution>()) {
            auto& prim_node = node->as<binary_convolution>();
            const auto& prim = prim_node.get_primitive();

            tensor specified_output_range(
                {0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1], prim->output_size.spatial[2]},
                1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range =
                calc_sliding_window_output_range<swor_mode::all>(prim_node.input().get_output_layout().size,
                                                                 filter_size,
                                                                 prim->input_offset,
                                                                 prim->stride,
                                                                 prim->dilation,
                                                                 true,
                                                                 1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        } else if (node->is_type<deconvolution>()) {
            auto& prim_node = node->as<deconvolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range(
                {0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1], prim->output_size.spatial[2]},
                1);

            auto filter_size = prim_node.weights(0).get_output_layout().size;

            auto calc_output_range = calc_sliding_window_needed_input_range(prim_node.input().get_output_layout().size,
                                                                            filter_size,
                                                                            prim->input_offset,
                                                                            prim->stride,
                                                                            {1, 1, 1, 1},
                                                                            true,
                                                                            1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        } else if (node->is_type<pooling>()) {
            auto& prim_node = node->as<pooling>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range(
                {0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1], prim->output_size.spatial[2]},
                1);

            // TODO: Check compatibility of output size calculation (with caffe).
            auto calc_output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
                prim_node.input().get_output_layout().size,
                prim->size,
                prim->input_offset,
                prim->stride,
                {1, 1, 1, 1},
                true,
                1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
    }

    return handling_needed;
}

// create new nodes for a program based on the set of nodes
// method created to be used by propagate_constants to build sub program from constant nodes
void program_impl::prepare_nodes(std::set<std::shared_ptr<program_node>> const& nodes) {
    for (const auto& itr : nodes) {
        if (itr.get()->is_type<data>()) {
            get_or_create(std::make_shared<input_layout>(itr.get()->id(),
                                                         itr.get()->as<data>().get_primitive()->mem.get_layout()));
        } else {
            get_or_create(itr->desc);
        }
    }
    for (const auto& node : nodes_map) {
        auto node_ptr = node.second;
        if (node_ptr == nullptr)
            throw std::runtime_error("NULL pointer in nodes_map.");
        // ToDo: avoid O(n^2) run time here (pass map instead of set?)
        bool found = false;
        for (const auto& src_node : nodes) {
            if (src_node == nullptr)
                throw std::runtime_error("NULL pointer in nodes_map.");
            if (node.first == src_node->get_primitive()->id) {
                copy_node_dependencies(node_ptr.get(), src_node.get());
                found = true;
                break;
            }
        }
        if (!found) {
            add_node_dependencies(node_ptr.get());
        }
        if (node_ptr->dependencies.size() == 0)
            inputs.push_back(node_ptr.get());
    }
}

// create all nodes from topology primitives, add dependencies among them and create inputs list
void program_impl::prepare_nodes(topology_impl const& topology) {
    auto const& topo_map = topology.get_primitives();
    for (const auto& prim : topo_map) {
        get_or_create(prim.second);
    }
    add_split_outputs();
    for (const auto& node : nodes_map) {
        auto node_ptr = node.second.get();
        if (node_ptr == nullptr)
            throw std::runtime_error("NULL pointer in nodes_map.");
        add_node_dependencies(node_ptr);
        if (node_ptr->dependencies.size() == 0) {
            inputs.push_back(node_ptr);
        }
    }
}

// add node's dependecies from its primitive dependencies
void program_impl::add_node_dependencies(program_node* node) {
    auto deps = node->get_primitive()->dependencies();
    // add pointers to node's dependencies
    for (auto& dep : deps) {
        try {
            auto dep_node = nodes_map.at(dep);
            node->dependencies.push_back(dep_node.get());
            dep_node->users.push_back(node);
        } catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + dep +
                                     " that is input to: " + node->get_primitive()->id);
        }
    }
}

/* helper method for program_impl constructor from list of nodes which
   copies src_node dependecies to the destination node dest_node dependencies.
   But only to those which appaer in this program implementation nodes_map */
void program_impl::copy_node_dependencies(program_node* dest_node, program_node* src_node) {
    if (dest_node->get_primitive()->id != src_node->get_primitive()->id) {
        throw std::runtime_error("Node " + src_node->get_primitive()->id + " and its copy " +
                                 dest_node->get_primitive()->id + " do not match.");
    }
    auto src_deps = src_node->get_dependencies();
    // add pointers to node's dependencies
    for (auto& src_dep : src_deps) {
        // do not copy dependencies to nodes which does not belong to the new (subgraph) topology
        if (nodes_map.find(src_dep->get_primitive()->id) == nodes_map.end())
            continue;

        try {
            auto dest_dep = nodes_map.at(src_dep->get_primitive()->id);
            dest_node->dependencies.push_back(dest_dep.get());
            dest_dep->users.push_back(dest_node);
        } catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + src_dep->get_primitive()->id +
                                     " that is input to: " + src_node->get_primitive()->id);
        }
    }
}

void program_impl::set_options() {
    static std::atomic<uint32_t> id_gen{0};
    prog_id = ++id_gen;
    assert(prog_id != 0);

    if ((options.get<build_option_type::tuning_config>()->config.mode == tuning_mode::tuning_tune_and_cache) &&
        !engine->configuration().enable_profiling) {
        throw std::invalid_argument("Engine must be created with profiling enabled in tune_and_cache mode!");
    }

    if (!options.get<build_option_type::force_implementations>()->forcing.empty()) {
        options.set_option(build_option::optimize_data(true));
    }
}

void program_impl::build_program(bool is_internal) {
    init_graph();
    { pre_optimize_graph(is_internal); }
    run_graph_compilation();
    { post_optimize_graph(is_internal); }
    prepare_memory_dependencies();
    engine->compile_program(*this);

    if (!is_internal)
        prim_info = get_current_stage_info();

    cleanup();
}

void program_impl::init_graph() {
    apply_opt_pass<graph_initializations>();

    apply_opt_pass<calculate_prior_boxes>();

    apply_opt_pass<mark_nodes>();
}

void program_impl::run_graph_compilation() { apply_opt_pass<compile_graph>(); }

void program_impl::pre_optimize_graph(bool is_internal) {
    // trim to outputs
    apply_opt_pass<trim_to_outputs>();  // ToDo remove hidden dependencies from trimm pass

    // handle symmetric and asymmetric padding for input
    apply_opt_pass<handle_input_padding>();

    // add reshape to input/parameters for some primitives
    apply_opt_pass<add_reshape_to_primitives>();

    processing_order.calculate_BFS_processing_order();  // this method makes sense only for OOOQ (out of order execution queue)

    apply_opt_pass<reverse_optional_nodes_outputs>();

    bool output_size_handling_enabled = analyze_output_size_handling_need();
    for (auto& node : processing_order) {
        if (!node->is_type<internal_primitive>() && !node->is_type<data>())
            node->get_output_layout();
    }

    if (options.get<build_option_type::optimize_data>()->enabled()) {
        apply_opt_pass<prepare_quantization>();
    }

    layout_optimizer lo(output_size_handling_enabled);
    set_layout_optimizer_attributes(lo);

    reorder_factory rf;
    if (options.get<build_option_type::optimize_data>()->enabled()) {
        apply_opt_pass<prepare_primitive_fusing>(lo);

        apply_opt_pass<reorder_inputs>(lo, rf);

        // TODO this code should be moved to post compilation after kernel selector will support handling reorder bias
        apply_opt_pass<pre_optimize_bias>(rf);

        // passes regarding conv + eltwise optimizations

        // shrinking eltwise if users are conv 1x1 with stride > 1 optimization
        apply_opt_pass<eltwise_shrinking>();

        // trying to set stride to 1x1 by shrinking convolutions before eltwise if doable
        apply_opt_pass<eltwise_remove_stride>();
    }

    apply_opt_pass<strided_slice_optimize>();

    apply_opt_pass<handle_reshape>();

    apply_opt_pass<prepare_padding>(output_size_handling_enabled);

    apply_opt_pass<remove_redundant_reorders>(lo, options.get<build_option_type::optimize_data>()->enabled());

    if (options.get<build_option_type::optimize_data>()->enabled()) {
        // Fuse conv + eltw after padding preparations
        apply_opt_pass<prepare_conv_eltw_fusing>(lo, lo.get_optimization_attributes().bfyx_f16_network);

        apply_opt_pass<prepare_conv_eltw_read_write_opt>();
    }

    apply_opt_pass<prepare_depthwise_sep_opt>();

    if (!is_internal) {
        // ToDo remove hidden dependencies from propagate_constants pass
        apply_opt_pass<propagate_constants>();
    }

    // try to fuse buffers (i.e. depth_concat in bfyx format) after padding calculations
    if (options.get<build_option_type::optimize_data>()->enabled()) {
        apply_opt_pass<prepare_buffer_fusing>();
    }

    // check if there exists some layout incompatibilities and add an reorder node if required
    apply_opt_pass<add_required_reorders>();
}

void program_impl::post_optimize_graph(bool is_internal) {
    // input reorder for fully connected if necessary
    apply_opt_pass<post_input_reorder>();

    reorder_factory rf;
    layout_optimizer lo;
    apply_opt_pass<post_optimize_weights>(rf);

    apply_opt_pass<remove_redundant_reorders>(lo, false, true);  // TODO: do we need it at this place also?

    if (!is_internal) {
        // ToDo remove hidden dependencies from propagate_constants pass
        apply_opt_pass<propagate_constants>();
    }

    apply_opt_pass<prep_opt_depthwise_sep_post>();
}

// mark if the node is constant assuming that all dependencies are marked properly
void program_impl::mark_if_constant(program_node& node) {
    if (node.get_dependencies().empty())
        return;
    if (node.is_type<prior_box>())
        return;
    node.constant = true;
    for (auto& dep : node.get_dependencies()) {
        if (!dep->constant) {
            node.constant = false;
            break;
        }
    }
}

// mark if the node is in data flow assuming that all dependencies are marked properly
void program_impl::mark_if_data_flow(program_node& node) {
    if (node.is_type<mutable_data>() || node.is_type<input_layout>()) {
        node.data_flow = true;
    } else {
        node.data_flow = false;
        size_t inputs_count = node.get_dependencies().size();
        if (node.is_type<detection_output>() || node.is_type<proposal>())
            inputs_count = 2;  // ignore third input as it is related to prior boxes (i.e. concat of prior-boxes)
        for (size_t idx = 0; idx < inputs_count; idx++) {
            if (node.get_dependency(idx).is_in_data_flow()) {
                node.data_flow = true;
                return;
            }
        }
    }
}

void program_impl::cleanup() {
    for (auto& node : processing_order)
        if (!node->is_type<internal_primitive>())
            node->get_output_layout();

    // in debug build, at the end, mark all nodes as outputs so user can query for buffers of all not-optimized nodes,
    // including internal ones etc.
    if (is_debug_build()) {
        for (auto& node : processing_order) {
            if (!node->is_output()) {
                node->set_output(true);
                outputs.push_back(node);
            }
        }
    }
}

void program_impl::add_split_outputs() {
    auto itr = nodes_map.begin();
    while (itr != nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        if (node->is_type<split>()) {
            auto split_prim = node->as<split>().typed_desc();
            primitive_id input_id = split_prim->input[0];
            auto split_num = split_prim->output_offsets.size();

            // create crop for each split ouptut provided
            for (decltype(split_num) i = 0; i < split_num; i++) {
                primitive_id output_id = node->id() + ":" + split_prim->output_ids[i];

                // create dummy crop primitive and add it to nodes map
                auto crop_prim =
                    std::make_shared<crop>(output_id, input_id, tensor{1, 1, 1, 1}, split_prim->output_offsets[i]);
                get_or_create(crop_prim);
            }
        }
    }
}

program_impl::nodes_ordering& program_impl::get_processing_order() { return processing_order; }

const program_impl::nodes_ordering& program_impl::get_processing_order() const { return processing_order; }

void add_memory_dependency(program_node* node, program_node* dep) {
    if (node->can_be_optimized() || !dep->can_be_optimized()) {
        node->add_memory_dependency(dep->id());
    } else {
        if (node->id() == dep->id()) {
            return;
        }
        for (auto subdep : dep->get_dependencies()) {
            add_memory_dependency(node, subdep);
            add_memory_dependency(subdep, node);
        }
    }
}

void program_impl::basic_memory_dependencies() {
    auto itr = processing_order.begin();
    std::vector<primitive_id> past_outputs;
    while (itr != processing_order.end()) {
        auto& node = *itr;
        itr++;

        // data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (auto it : node->get_dependencies()) {
            add_memory_dependency(node, it);
            add_memory_dependency(it, node);
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs,
        // this output has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output())
            past_outputs.push_back(node->id());
    }
}

void program_impl::skipped_branch_memory_dependencies() {
    // Primitive A can't use primitive B buffer if processing_num(B) < processing_num(A) and for any usr - the user of B
    // processing_num(usr) > processing_num(A) Otherwise it could override data that has to be used in the future.
    auto itrB = processing_order.begin();
    while (itrB != processing_order.end()) {
        auto& nodeB = *itrB;
        auto itrA = ++itrB;
        if (nodeB->get_users().size() == 0)
            continue;

        // find the last user of B in processing order
        auto itrUsr = nodeB->get_users().begin();
        auto lastUsr = itrUsr++;
        while (itrUsr != nodeB->get_users().end()) {
            if (processing_order.get_processing_number(*lastUsr) < processing_order.get_processing_number(*itrUsr))
                lastUsr = itrUsr;
            itrUsr++;
        }

        // mark all nodes in between B and lastUsr of B as forbidden to share buffer with B
        while (itrA != processing_order.get_processing_iterator(**lastUsr) && itrA != processing_order.end()) {
            auto& nodeA = *itrA;
            itrA++;
            add_memory_dependency(nodeA, nodeB);
            add_memory_dependency(nodeB, nodeA);
        }
    }
}

void program_impl::oooq_memory_dependencies() {
    // For oooq memory dependencies nodes A and B can't share memory if
    // processing_num(A) < processing_num(B) and there is no path from A to B.
    // Assuming precalculation of reachability this function has complexity O(N^2 log N).

    // First create transitive closure of the graph,
    // giving us mapping of node to set of all users that can be reached from this node.

    // Start with map generated from direct users.
    auto user_map = std::map<program_node*, std::set<program_node*>>();
    for (auto node : get_processing_order()) {
        user_map[node].insert(node->get_users().begin(), node->get_users().end());
    }

    // Iteratively extend the users set by adding closure over existing users untill no change occurs.
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& node_users : user_map) {
            auto node = node_users.first;
            const auto& users = node_users.second;

            for (auto user : users) {
                for (auto user_users : user_map[user]) {
                    if (user_map[node].insert(user_users).second)
                        changed = true;
                }
            }
        }
    }

    // Connection query:
    auto are_connected = [&](program_node* A, program_node* B) { return user_map.at(A).count(B) == 1; };

    auto itr_A = processing_order.begin();

    while (itr_A != processing_order.end()) {
        auto itr_B = ++itr_A;
        while (itr_B != processing_order.end()) {
            if (!are_connected(*itr_A, *itr_B)) {
                add_memory_dependency(*itr_A, *itr_B);
                add_memory_dependency(*itr_B, *itr_A);
            }
            itr_B++;
        }
    }
}

void program_impl::prepare_memory_dependencies() {
    if (!get_engine().configuration().enable_memory_pool)
        return;

    basic_memory_dependencies();
    skipped_branch_memory_dependencies();
    oooq_memory_dependencies();
}

std::string program_impl::get_memory_dependencies_string() const {
    std::string mem_dep = "Memory dependencies/restrictions:\n";
    auto itr = processing_order.begin();
    while (itr != processing_order.end()) {
        auto& node = *itr;
        itr++;
        mem_dep = mem_dep.append("primitive: ").append(node->id()).append(" restricted list: ");
        for (auto it : node->get_memory_dependencies())
            mem_dep = mem_dep.append(it).append(", ");
        mem_dep = mem_dep.append("\n");
    }
    return mem_dep;
}

void program_impl::apply_needed_padding(program_node& node, program_node& prev_node, const padding& needed_padding) {
    auto target_layout = prev_node.get_output_layout();

    // Short circuit if padding did not change.
    if (target_layout.data_padding == needed_padding)
        return;

    // Special handling for input nodes.
    if (prev_node.is_type<input_layout>() || prev_node.is_type<mutable_data>()) {
        target_layout.data_padding = needed_padding;

        auto r_prim = std::make_shared<reorder>("reorder_input_" + node.id(), prev_node.id(), target_layout);
        add_intermediate(r_prim, node, 0);
        return;
    }

    prev_node.merge_output_padding(needed_padding);
}

void program_impl::reverse_connection(program_node& dep_node, program_node& user_node) {
    if (std::find(dep_node.users.begin(), dep_node.users.end(), &user_node) != dep_node.users.end()) {
        remove_connection(dep_node, user_node);
        add_connection(user_node, dep_node);
    } else {
        throw std::runtime_error("Trying to reverse connection, but nodes are wrongly or not connected.");
    }
}

program_node& program_impl::get_or_create(std::shared_ptr<primitive> prim) {
    auto itr = nodes_map.lower_bound(prim->id);
    if (itr != nodes_map.end() && itr->first == prim->id)
        return *itr->second;

    auto new_node = prim->type->create_node(*this, prim);
    nodes_map.insert(itr, {prim->id, new_node});
    return *new_node;
}

void program_impl::add_intermediate(program_node& node,
                                    program_node& next,
                                    size_t prev_idx,
                                    bool connect_int_node_with_old_dep,
                                    bool move_usrs_of_prev_to_node) {
    if (connect_int_node_with_old_dep && !node.dependencies.empty())
        throw std::invalid_argument(
            "Node which is about to be added in between two other nodes should not have any existing dependencies");

    auto& prev = next.get_dependency(prev_idx);
    // firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    if (connect_int_node_with_old_dep) {
        add_connection(prev, node);
        if (processing_order.size() != 0) {
            processing_order.insert_next(&prev, &node);
        }
    }

    if (move_usrs_of_prev_to_node) {
        auto itr = prev.get_users().begin();
        while (itr != prev.get_users().end()) {
            auto usr = *itr;
            itr++;
            if (usr->id() != node.id())
                usr->replace_dependency(prev, node);
        }
        mark_if_constant(prev);
        mark_if_constant(node);
        mark_if_data_flow(prev);
        mark_if_data_flow(node);
    } else {
        next.replace_dependency(prev_idx, node);
        node.constant = prev.constant;
        node.data_flow = prev.data_flow;
    }
}

void program_impl::add_intermediate(std::shared_ptr<primitive> prim,
                                    program_node& next,
                                    size_t prev_idx,
                                    bool connect_int_node_with_old_dep,
                                    bool move_usrs_of_prev_to_node) {
    add_intermediate(get_or_create(prim), next, prev_idx, connect_int_node_with_old_dep, move_usrs_of_prev_to_node);
}

void program_impl::add_intermediate(program_node& node,
                                    program_node& next,
                                    program_node& prev,
                                    bool connect_int_node_with_old_dep,
                                    bool move_usrs_of_prev_to_node) {
    bool node_found = false;
    size_t idx = 0;
    for (size_t i = 0; i < next.get_dependencies().size(); i++) {
        auto& input = next.get_dependency(i);
        if (input.id() == prev.id()) {
            idx = i;
            node_found = true;
            break;
        }
    }
    if (!node_found) {
        throw std::runtime_error("Trying to add intermediate node in between " + next.id() + " and dependecy " + prev.id() +
                        " but they are not connected in this way.");
    }
    add_intermediate(node, next, idx, connect_int_node_with_old_dep, move_usrs_of_prev_to_node);
}

void program_impl::add_connection(program_node& prev, program_node& next) {
    prev.users.push_back(&next);
    next.dependencies.push_back(&prev);
}

void program_impl::remove_connection(program_node& prev, program_node& next) {
    prev.users.remove(&next);
    next.dependencies.erase(std::remove(next.dependencies.begin(), next.dependencies.end(), &prev),
                            next.dependencies.end());
}

void program_impl::remove_all_connections(program_node& node) {
    // since the graph is not topological sorted, we need to remove the node from both dependencies and users
    for (auto& e : node.users) {
        e->dependencies.erase(std::remove(e->dependencies.begin(), e->dependencies.end(), &node),
                              e->dependencies.end());
    }
    for (auto& e : node.dependencies) {
        e->users.remove(&node);
    }
    node.dependencies.clear();
    node.users.clear();
}

void program_impl::rename(program_node& node, primitive_id const& new_id) {
    if (nodes_map.count(new_id))
        throw std::runtime_error("Trying to rename program_node but node with id " + new_id + " already exists");
    if (node.is_output())
        throw std::invalid_argument(
            "Trying to rename an output node. If you intend to do that, please clear 'output' flag manually.");

    auto node_itr = nodes_map.find(node.id());
    if (node_itr == nodes_map.end()) return;

    auto node_ptr = node_itr->second;
    nodes_map.emplace(new_id, node_ptr);
    nodes_map.erase(node.id());

    if (!node.is_type<internal_primitive>())
        const_cast<primitive_id&>(node.desc->id) = new_id;
    else
        reinterpret_cast<details::internal_program_node_base&>(node).internal_id = new_id;
}

void program_impl::swap_names(program_node& node1, program_node& node2) {
    const auto _extract_id = [](program_node& node) -> primitive_id& {
        if (!node.is_type<internal_primitive>())
            return const_cast<primitive_id&>(node.desc->id);
        else
            return reinterpret_cast<details::internal_program_node_base&>(node).internal_id;
    };

    nodes_map.at(node1.id()).swap(nodes_map.at(node2.id()));
    std::swap(_extract_id(node1), _extract_id(node2));
}

void program_impl::replace_all_usages(program_node& old_node, program_node& new_node) {
    auto itr = old_node.users.begin();
    auto cnt = old_node.users.size();
    while (cnt != 0) {
        cnt--;
        auto& usage = (*itr++);
        usage->replace_dependency(old_node, new_node);
    }
}

void program_impl::replace(program_node& old_node, program_node& new_node) {
    if (!new_node.dependencies.empty() || !new_node.users.empty())
        throw std::invalid_argument("Node which is about to replace other node should be detached");

    if (new_node.is_output())
        throw std::invalid_argument(
            "Replacement node shouldn't be marked as an output since it's impossible to rename such node.");

    auto id = old_node.id();
    new_node.output_layout = old_node.get_output_layout();
    new_node.valid_output_layout = old_node.valid_output_layout;

    // copy old's dependencies
    while (!old_node.dependencies.empty()) {
        auto& dep = old_node.dependencies.front();
        add_connection(*dep, new_node);
        remove_connection(*dep, old_node);
    }

    // append users
    for (auto& user : old_node.users) {
        new_node.users.push_back(user);
        for (auto& users_dep : user->dependencies) {
            if (users_dep == &old_node) {
                users_dep = &new_node;
                break;
            }
        }
    }

    old_node.users.clear();

    bool old_was_output = false;
    // copy node's state
    if (old_node.is_output()) {
        old_was_output = true;
        old_node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &old_node), outputs.end());
    }
    if (new_node.is_input())
        inputs.push_back(&new_node);
    if (old_node.is_input())
        inputs.remove(&old_node);

    new_node.constant = old_node.constant;
    new_node.data_flow = old_node.data_flow;
    new_node.user_mark = old_node.user_mark;

    processing_order.insert(&old_node, &new_node);
    if (processing_order.get_processing_iterator(old_node) != processing_order.end())
        processing_order.erase(&old_node);
    nodes_map.erase(id);
    rename(new_node, id);

    // mark new node as an output after renaming
    if (old_was_output) {
        new_node.set_output(true);
        outputs.push_back(&new_node);
    }
}

bool program_impl::remove_if_dangling(program_node& node) {
    if (!node.users.empty())
        return false;
    if (!node.dependencies.empty())
        return false;

    if (!node.is_output() || is_debug_build()) {
        if (node.is_input())
            inputs.remove(&node);

        if (std::find(processing_order.begin(), processing_order.end(), &node) != processing_order.end())
            processing_order.erase(&node);
        optimized_out.push_back(node.id());
        nodes_map.erase(node.id());
    }
    return true;
}

bool program_impl::extract_and_remove(program_node& node) {
    if (node.get_dependencies().size() != 1)
        return false;

    if (node.is_output() && !is_debug_build()) {
        auto& prev = node.get_dependency(0);
        auto node_id = node.id();

        node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &node), outputs.end());

        rename(node, "_cldnn_tmp_" + node_id);
        rename(prev, node_id);

        prev.set_output(true);
        outputs.push_back(&prev);
    }

    auto& input = node.get_dependency(0);
    node.dependencies.clear();
    input.users.remove(&node);

    if (!node.is_endpoint())
        replace_all_usages(node, input);
    else
        remove_if_dangling(node);

    return true;
}

void program_impl::fuse_nodes(program_node &fused_node, program_node &peer_node) {
    auto peer_layout = peer_node.get_output_layout();
    fused_primitive_desc local_desc;
    local_desc.node = get_node_ptr(peer_node.id());
    local_desc.dep_start_idx = fused_node.get_dependencies().size();
    local_desc.output_layout = peer_layout;
    local_desc.activation = activation_func::none;
    if (!peer_node.get_fused_activations_funcs().empty()) {
        if (peer_node.get_fused_activations_funcs().size() > 1)
            CLDNN_ERROR_MESSAGE(peer_node.id(), "Fused primitive descriptor doesn't support > 1 activation functions in a peer node");

        local_desc.activation = peer_node.get_fused_activations_funcs()[0];
        local_desc.activation_params = peer_node.get_fused_activations_params()[0];
    }

    cldnn::padding needed_padding = padding::max(peer_layout.data_padding,
                                                 fused_node.get_output_layout().data_padding);

    // Add new dependencies to the fused_node
    for (size_t i = 0; i < peer_node.get_dependencies().size(); i++) {
        auto& dep = peer_node.get_dependency(i);
        if (dep.id() == fused_node.id())
            continue;
        fused_node.dependencies.push_back(&dep);
        local_desc.deps.push_back(dep.id());
        dep.users.push_back(&fused_node);
    }
    fused_node.add_fused_primitive(local_desc);
    // This shouldn't happen, but who knows...
    if (peer_node.has_fused_primitives()) {
        fused_node.add_fused_primitives(peer_node.get_fused_primitives());
    }
    add_optimized_primitive_info(peer_node.id(), { fused_node.id() });

    // Remove all edges connected with peer node
    while (peer_node.get_dependencies().size() > 0) {
        auto& dep = peer_node.get_dependency(peer_node.get_dependencies().size() - 1);
        remove_connection(dep, peer_node);
    }
    replace_all_usages(peer_node, fused_node);

    // Update output layout. Recalculation is not needed.
    fused_node.merge_output_padding(needed_padding);
    fused_node.set_output_layout(peer_layout, false);
    fused_node.recalc_output_layout(true);
}

void program_impl::remove_nodes(std::list<program_node*>& to_remove) {
    for (auto const& node : to_remove) {
        if (node->is_input()) {
            get_inputs().remove(node);
        } else {
            for (auto& dep : node->dependencies) dep->users.remove(node);
        }
        for (auto& user : node->users) {
            user->dependencies.erase(std::remove(user->dependencies.begin(), user->dependencies.end(), node),
                                     user->dependencies.end());
        }
        get_processing_order().erase(node);
        optimized_out.push_back(node->id());
        nodes_map.erase(node->id());
    }
}

void program_impl::dump_memory_pool() const {
    if (!get_engine().configuration().enable_memory_pool)
        return;
    auto path = get_dir_path(options);
    if (path.empty()) {
        return;
    }
    path += "cldnn_memory_pool.log";
    auto dep = get_memory_dependencies_string();
    get_engine().dump_memory_pool(*this, path, dep);
    std::string dump_file_name;
    if (pm->get_pass_count() < 10)
        dump_file_name += "0";
    dump_file_name += std::to_string(pm->get_pass_count()) + "_memory_pool";
    pm->inc_pass_count();
    dump_program(dump_file_name.c_str(), true);
}

// TODO: break this function into number of smaller ones + add per-primitive fields (possibly use
// primitive_inst::to_string?)
void program_impl::dump_program(const char* stage,
                                bool with_full_info,
                                std::function<bool(program_node const&)> const& filter) const {
    std::string path = get_dir_path(options);
    if (path.empty() || !with_full_info) {
        return;
    }

    std::ofstream graph(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".graph");
    dump_graph_init(graph, *this, filter);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".info");
    dump_graph_info(graph, *this, filter);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".order");
    dump_graph_processing_order(graph, *this);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".optimized");
    dump_graph_optimized(graph, *this);
}

program_impl::primitives_info program_impl::get_current_stage_info() const {
    primitives_info info;
    // Get info for actually executed graph nodes
    int exec_id = 0;
    for (auto& p : get_processing_order()) {
        std::vector<primitive_id> users;
        for (auto& user : p->users) {
            users.push_back(user->id());
        }
        std::vector<primitive_id> dependencies;
        for (auto& a : p->dependencies) {
            dependencies.push_back(a->id());
        }

        std::vector<primitive_id> fused;
        for (auto& op_prim : optimized) {
            for (auto& fused_to : op_prim.second) {
                if (p->id() == fused_to) {
                    fused.push_back(op_prim.first);
                }
            }
        }

        primitive_info pi(p->id(),
                          type_to_str(p->get_primitive()),
                          dependencies,
                          users,
                          fused,
                          p->get_output_layout(),
                          fmt_to_str(p->get_output_layout().format),
                          p->selected_impl ? p->selected_impl->get_kernel_name() : "",
                          p->selected_impl ? p->selected_impl->is_cpu() : false,
                          exec_id++);

        info.push_back(pi);
    }

    return info;
}

void program_impl::save_pass_info(std::string pass_name) {
    // TODO: Directory path here can be probably changed to some bool flag
    if (!options.get<build_option_type::graph_dumps_dir>()->directory_path.empty())
        optimizer_passes_info.emplace_back(pass_name, get_current_stage_info());
}

const program_impl::graph_optimizer_info& program_impl::get_optimizer_passes_info() const {
    return optimizer_passes_info;
}

const program_impl::primitives_info& program_impl::get_primitives_info() const { return prim_info; }

void program_impl::apply_opt_pass(base_pass& pass) { pm->run(*this, pass); }

void program_impl::set_layout_optimizer_attributes(layout_optimizer& lo) {
    lo.set_implementation_forcing(options.get<build_option_type::force_implementations>()->forcing);

    // first pass to set layout optimization_attributes for topology
    bool can_use_f16 = true;
    size_t total_conv_layers = 0;
    size_t total_asym_quantized_conv_layers = 0;
    size_t total_dw_conv_layers = 0;
    size_t total_dw_splitted_conv_layers = 0;
    size_t total_1x1_fm_conv_layers = 0;
    size_t total_grouped_conv_layers = 0;
    size_t opt_conv_layers_bfyx_f16 = 0;
    size_t opt_conv_layers_bfzyx_f16 = 0;
    size_t opt_deconv_layers_bfzyx_f16 = 0;

    for (auto& node : get_processing_order()) {
        auto& prim = *node;
        if (prim.type() == cldnn::convolution::type_id()) {
            auto& conv = prim.as<convolution>();
            if (conv.get_primitive()->split() > 1)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::splitted_convolution, 1);

            if (conv.get_primitive()->groups > 1)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::group_convolution, 1);

            if (conv.get_primitive()->deformable_mode)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::deformable_convolution, 1);

            auto input_size = node->get_dependency(0).get_output_layout().size;
            auto ifm = static_cast<uint32_t>(input_size.feature[0]);
            if (conv.get_primitive()->groups == ifm)
                total_dw_conv_layers++;
            else if (static_cast<uint32_t>(node->as<convolution>().get_split()) == ifm)
                total_dw_splitted_conv_layers++;  // this counter is needed due to compatibility with bfyx_f16 heuristics
            else if (conv.get_primitive()->groups > 1 || conv.get_primitive()->split() > 1)
                total_grouped_conv_layers++;

            if (input_size.spatial[0] == 1 && input_size.spatial[1] == 1)
                total_1x1_fm_conv_layers++;

            if (lo.is_format_optimized(conv, format::bfyx_f16))
                opt_conv_layers_bfyx_f16++;

            if (lo.is_format_optimized(conv, format::bfzyx_f16))
                opt_conv_layers_bfzyx_f16++;

            if (conv.weights_zero_points_term() || conv.activations_zero_points_term())
                total_asym_quantized_conv_layers++;

            total_conv_layers++;
        }
        if (prim.type() == cldnn::deconvolution::type_id()) {
            if (lo.is_format_optimized(prim.as<deconvolution>(), format::bfzyx_f16))
                opt_deconv_layers_bfzyx_f16 += 1;
        }

        // list of layers that do not support yxfb or perform worse than bfyx
        if (prim.type() == cldnn::detection_output::type_id() || prim.type() == cldnn::proposal::type_id() ||
            prim.type() == cldnn::roi_pooling::type_id() || prim.type() == cldnn::deconvolution::type_id() ||
            prim.type() == cldnn::resample::type_id() || prim.type() == cldnn::reorg_yolo::type_id())
            lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bfyx_only_layer, 1);

        if (prim.is_in_data_flow() &&
            prim.type() != cldnn::convolution::type_id() &&
            prim.type() != cldnn::activation::type_id() &&
            prim.type() != cldnn::pooling::type_id() &&
            prim.type() != cldnn::eltwise::type_id() &&
            prim.type() != cldnn::permute::type_id() &&
            prim.type() != cldnn::reshape::type_id() &&
            prim.type() != cldnn::detection_output::type_id() &&
            prim.type() != cldnn::binary_convolution::type_id() &&
            prim.type() != cldnn::quantize::type_id() &&
            prim.type() != cldnn::custom_gpu_primitive::type_id() &&
            prim.type() != cldnn::concatenation::type_id() &&
            prim.type() != cldnn::fully_connected::type_id() &&
            prim.type() != cldnn::reorder::type_id() &&
            prim.type() != cldnn::input_layout::type_id() &&
            prim.type() != cldnn::softmax::type_id() &&
            prim.type() != cldnn::prior_box::type_id() &&
            prim.type() != cldnn::resample::type_id() &&
            prim.type() != cldnn::crop::type_id() &&
            prim.type() != cldnn::scale::type_id())
            can_use_f16 = false;

        // WA to keep bfyx_f16 layout disabled for some topologies where it leads to regressions.
        // Detects if given crop layer is located in the very beginning of the graph.
        if (prim.type() == cldnn::crop::type_id()) {
            if (!prim.get_dependencies()[0]->is_type<reorder>() || !prim.get_dependencies()[0]->get_dependencies()[0]->is_input())
                can_use_f16 = false;
        }
    }

    // Due to fact that single winograd convolution is faster than bfyx_f16 and
    // using them together leads do redundant reorders, whole topology switch
    // will be performed if at least half of layers can use bfyx_f16.
    const float cond_denom = total_conv_layers > 0 ? 1.0f / static_cast<float>(total_conv_layers) : 1.0f;

    bool should_use_bfyx_f16_conv = can_use_f16 &&
                                    total_conv_layers > 11 &&
                                    opt_conv_layers_bfyx_f16 * cond_denom > 0.5f &&
                                    total_grouped_conv_layers == 0 &&
                                    total_dw_splitted_conv_layers == 0;  // conv with groups are not supported correctly yet

    bool should_use_fs_b_yx_fsv32_conv = total_conv_layers > 11 &&
                                         total_grouped_conv_layers == 0 &&
                                         total_1x1_fm_conv_layers * cond_denom < 0.8f;

    bool should_use_b_fs_zyx_fsv32_conv = total_asym_quantized_conv_layers > 1;

    if (should_use_fs_b_yx_fsv32_conv)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::fs_b_yx_fsv32_network, 1);

    if (should_use_b_fs_zyx_fsv32_conv)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::b_fs_zyx_fsv32_network, 1);

    if (should_use_bfyx_f16_conv)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bfyx_f16_network, 1);

    if (opt_conv_layers_bfzyx_f16 >= 1 || opt_deconv_layers_bfzyx_f16 >= 1)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bfzyx_f16_network, 1);
}
