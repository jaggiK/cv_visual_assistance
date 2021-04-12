/*
// Copyright (c) 2018-2019 Intel Corporation
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

#pragma once

#include "program_impl.h"
#include "layout_optimizer.h"
#include <string>
#include <vector>
#include <memory>
#include <list>
#include <utility>
#include <set>

#include <fstream>

namespace cldnn {
class base_pass {
    friend class pass_manager;

public:
    explicit base_pass(const std::string& pass_name) : name(pass_name) {}
    virtual void run(program_impl& p) = 0;
    std::string get_name() { return name; }
    void clean_marks(program_impl& p) {
        for (auto& node : p.get_processing_order()) {
            node->unmark();
        }
    }

private:
    const std::string name;
};

class pass_manager {
public:
    explicit pass_manager(program_impl& p);
    void run(program_impl& p, base_pass& pass);
    uint32_t get_pass_count() { return pass_count; }
    uint32_t inc_pass_count() { return ++pass_count; }
    ~pass_manager() {}

private:
    uint32_t pass_count;
    std::ofstream graph_opt_log;
};

class add_required_reorders : public base_pass {
public:
    add_required_reorders() : base_pass("add_required_reorders") {}

private:
    void run(program_impl& p) override;
    void add_reorder(program_impl& p, program_node* node, program_node* usr, const layout& reorder_layout);
};

class add_reshape_to_primitives : public base_pass {
public:
    add_reshape_to_primitives() : base_pass("add_reshape_to_primitives_pass") {}

private:
    void run(program_impl& p) override;
};

class calculate_prior_boxes : public base_pass {
public:
    calculate_prior_boxes() : base_pass("calculated_prior_boxes") {}

private:
    void run(program_impl& p) override;
};

class compile_graph : public base_pass {
public:
    compile_graph() : base_pass("compile_graph") {}

private:
    void run(program_impl& p) override;
};

class eltwise_shrinking : public base_pass {
public:
    eltwise_shrinking() : base_pass("eltwise_shrinking") {}

private:
    void run(program_impl& p) override;
};

class eltwise_remove_stride : public base_pass {
public:
    eltwise_remove_stride() : base_pass("eltwise_remove_stride") {}

private:
    void run(program_impl& p) override;
    void conv_stride_extend(program_impl& p, program_node& node, cldnn::tensor& tensor);
};

class graph_initializations : public base_pass {
public:
    graph_initializations() : base_pass("init") {}

private:
    void run(program_impl& p) override;
    void replace_nodes(program_impl& p);
    void handle_detection_output(program_impl& p);
    void handle_lstm(program_impl& p);
    void handle_dynamic_lstm(program_impl& p);
    void set_outputs(program_impl& p);
};

class handle_reshape : public base_pass {
public:
    handle_reshape() : base_pass("handle_reshape") {}

private:
    void run(program_impl& p) override;
};

class handle_input_padding : public base_pass {
public:
    handle_input_padding() : base_pass("handle_input_padding") {}

private:
    void run(program_impl& p) override;
};

class mark_nodes : public base_pass {
public:
    mark_nodes() : base_pass("analyzed_graph") {}

private:
    void run(program_impl& p) override;
    void mark_constants(program_impl& p);
    void mark_data_flow(program_impl& p);
};

class prepare_buffer_fusing : public base_pass {
public:
    prepare_buffer_fusing() : base_pass("prepare_buffer_fusing") {}

private:
    void run(program_impl& p) override;
};

class prepare_quantization : public base_pass {
public:
    prepare_quantization() : base_pass("prepare_quantization") {}

private:
    void run(program_impl& p) override;
    void prepare_packed_quantize(program_impl& p);
    void prepare_scale_shift_opt(program_impl& p);
    void remove_fake_reorders(program_impl& p);
    void prepare_asymmetric_quantization(program_impl& p);
};

class prepare_conv_eltw_fusing : public base_pass {
public:
    explicit prepare_conv_eltw_fusing(layout_optimizer& lo_ref, bool bfyx_f16_opt = false) :
        base_pass("prepare_conv_eltw_fusing"), _lo(lo_ref), bfyx_f16_opt(bfyx_f16_opt) {}

private:
    void run(program_impl& p) override;
    void fuse_conv_eltwise(program_impl& p, program_node* node);
    layout_optimizer& _lo;
    bool bfyx_f16_opt;
};

class prepare_conv_eltw_read_write_opt : public base_pass {
public:
    prepare_conv_eltw_read_write_opt() : base_pass("prepare_conv_eltw_read_write_opt") {}

private:
    void run(program_impl& p) override;
    void conv_eltwise_read_write_opt(program_impl& p, program_node* node);
};

class prepare_depthwise_sep_opt : public base_pass {
public:
    prepare_depthwise_sep_opt() : base_pass("prepare_depthwise_sep_opt") {}

private:
    void run(program_impl& p) override;
    template <typename T>
    void optimize_depthwise_sep_pre(T& node);
};

class prep_opt_depthwise_sep_post : public base_pass {
public:
    prep_opt_depthwise_sep_post() : base_pass("prep_opt_depthwise_sep_post") {}

private:
    void run(program_impl& p) override;
    template <typename T>
    void optimize_depthwise_sep_pre(program_impl& p, T& node);
};

class prepare_primitive_fusing : public base_pass {
public:
    explicit prepare_primitive_fusing(layout_optimizer& lo_ref) :
        base_pass("prepare_primitive_fusing"), _lo(lo_ref) {}

private:
    void run(program_impl& p) override;
    void fuse_reorders(program_impl& p);
    void fuse_activations(program_impl& p);
    void fuse_skip_layers(program_impl& p);
    void fuse_simple_primitives(program_impl &p);
    layout_optimizer& _lo;
};

class pre_optimize_bias : public base_pass {
public:
    explicit pre_optimize_bias(reorder_factory& rf_ref);

private:
    void run(program_impl& p) override;
    virtual void run(program_impl& p, reorder_factory& rf);
    template <typename T>
    void optimize_bias(T& node, reorder_factory& rf, program_impl& p);
    reorder_factory& _rf;
};

class prepare_padding : public base_pass {
public:
    explicit prepare_padding(bool output_size_handling_enabled_switch)
        : base_pass("prepare_padding"), output_size_handling_enabled(output_size_handling_enabled_switch) {}

private:
    void run(program_impl& p) override;
    bool output_size_handling_enabled;
};

class post_input_reorder : public base_pass {
public:
    post_input_reorder() : base_pass("post_input_reorder") {}

private:
    void run(program_impl& p) override;
    program_node& add_reorder(program_impl& p, program_node* node, program_node* usr, const layout& reorder_layout);
};

class post_optimize_weights : public base_pass {
public:
    explicit post_optimize_weights(reorder_factory& rf_ref);

private:
    struct weights_bias_offset {
        size_t weights_offset;
        size_t bias_offset;

        // When using this ctor weights offset is added to the bias_offset
        weights_bias_offset(const size_t w_offset, const size_t b_offset)
            : weights_offset(w_offset)
            , bias_offset(weights_offset + b_offset)
        {}
    };

    void run(program_impl& p) override;
    template<typename T>
    weights_bias_offset get_weights_bias_offset(const T& node);
    template<typename T>
    void optimize_weights(T& node, program_impl& p);
    reorder_factory& _rf;
};

class propagate_constants : public base_pass {
public:
    propagate_constants() : base_pass("propagate_constants") {}

private:
    void run(program_impl& p) override;
    std::list<std::pair<primitive_id, memory_impl::ptr>> calculate(engine_impl& engine, build_options bo);
    bool has_non_const_user(program_node& node) const;
    void handle_constant(program_impl& prog, program_node& node);
    void add_constant(program_impl& prog, program_node& node);
    void add_deps_to_tpl(program_impl& prog, const std::vector<program_node*>& node);

    bool has_non_trivial_constants = false;
    std::list<typed_program_node<data>*> const_inputs;
    std::vector<primitive_id> const_outputs;
    std::set<std::shared_ptr<program_node>> nodes;
};

class remove_redundant_reorders : public base_pass {
public:
    explicit remove_redundant_reorders(layout_optimizer& lo_ref, bool enable_reorder_fusing = false, bool update_implementations = false);
    void run(program_impl& p) override;

private:
    layout_optimizer& lo;
    bool enable_reorder_fusing;
    bool update_implementations;
};

class reorder_inputs : public base_pass {
public:
    reorder_inputs(layout_optimizer& lo_ref, reorder_factory& rf_ref);

private:
    void run(program_impl& p) override;
    virtual void run(program_impl& p, layout_optimizer& lo, reorder_factory& rf);
    layout_optimizer& _lo;
    reorder_factory& _rf;
};

class trim_to_outputs : public base_pass {
public:
    trim_to_outputs() : base_pass("trimmed") {}

private:
    void run(program_impl& p) override;
};

class strided_slice_optimize : public base_pass {
public:
    strided_slice_optimize() : base_pass("strided_slice_optimize") {}
    void run(program_impl& p) override;
};

class reverse_optional_nodes_outputs : public base_pass {
public:
    reverse_optional_nodes_outputs() : base_pass("reverse_optional_nodes_outputs") {}
    void run(program_impl& p) override;
};
}  // namespace cldnn
