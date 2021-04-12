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
#include "api/binary_convolution.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<binary_convolution> : public typed_program_node_base<binary_convolution> {
    using parent = typed_program_node_base<binary_convolution>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog), split(this->get_primitive()->split()), depthwise_sep_opt(false) {}

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    program_node& input() const { return get_dependency(0); }

    program_node& weights(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx);
    }

    // Define bias functions to be able to reuse get_weights_bias_default_params<T> function
    program_node& bias(size_t /*idx*/ = 0) const {
        throw std::runtime_error("bias should not be used in binary convolution");
    }

    bool bias_term() const { return false; }

private:
    int32_t split;
    bool depthwise_sep_opt;
};

using binary_convolution_node = typed_program_node<binary_convolution>;

template <>
class typed_primitive_inst<binary_convolution> : public typed_primitive_inst_base<binary_convolution> {
    using parent = typed_primitive_inst_base<binary_convolution>;

public:
    static layout calc_output_layout(binary_convolution_node const& node);

    static std::string to_string(binary_convolution_node const& node);

public:
    typed_primitive_inst(network_impl& network, binary_convolution_node const& node);

    memory_impl& weights_memory(size_t index) const {
        if (static_cast<int32_t>(index) >= node.get_split())
            throw std::range_error("weights offset too big");

        return dep_memory(1 + index);
    }
};

using binary_convolution_inst = typed_primitive_inst<binary_convolution>;

}  // namespace cldnn
