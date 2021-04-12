/*
// Copyright (c) 2016 Intel Corporation
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
#include "api/activation.hpp"
#include "primitive_inst.h"
#include "kernel_selector/core/actual_kernels/activation/activation_kernel_base.h"
#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<activation> : public typed_program_node_base<activation> {
    using parent = typed_program_node_base<activation>;
    typed_program_node(const std::shared_ptr<activation> prim, program_impl& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& slope_input() const { return get_dependency(1); }

    bool is_parameterized() const { return !typed_desc()->additional_params_input.empty(); }

    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        kernel_selector::base_activation_params p;
        p.function = get_kernel_selector_activation_param(typed_desc()->activation_function);
        p.m = typed_desc()->additional_params.a;
        p.n = typed_desc()->additional_params.b;
        return std::make_shared<kernel_selector::activation_fuse_params>(p);
    }
};

using activation_node = typed_program_node<activation>;

template <>
class typed_primitive_inst<activation> : public typed_primitive_inst_base<activation> {
    using parent = typed_primitive_inst_base<activation>;

public:
    static layout calc_output_layout(activation_node const& node);
    static std::string to_string(activation_node const& node);

public:
    typed_primitive_inst(network_impl& network, activation_node const& node);

    memory_impl& slope_memory() const { return dep_memory(1); }

    bool is_parameterized() const { return !argument.additional_params_input.empty(); }
};

using activation_inst = typed_primitive_inst<activation>;
}  // namespace cldnn
