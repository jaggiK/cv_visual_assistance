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
#include "api/fully_connected_grad_input.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<fully_connected_grad_input> : public typed_program_node_base<fully_connected_grad_input> {
    using parent = typed_program_node_base<fully_connected_grad_input>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& weights() const { return get_dependency(2); }
};

using fully_connected_grad_input_node = typed_program_node<fully_connected_grad_input>;

template <>
class typed_primitive_inst<fully_connected_grad_input> : public typed_primitive_inst_base<fully_connected_grad_input> {
    using parent = typed_primitive_inst_base<fully_connected_grad_input>;

public:
    static layout calc_output_layout(fully_connected_grad_input_node const& node);
    static std::string to_string(fully_connected_grad_input_node const& node);

public:
    typed_primitive_inst(network_impl& network, fully_connected_grad_input_node const& node);

    memory_impl& weights_memory() const { return dep_memory(2); }
    bool bias_term() const { return false; }
};

using fully_connected_grad_input_inst = typed_primitive_inst<fully_connected_grad_input>;

}  // namespace cldnn
