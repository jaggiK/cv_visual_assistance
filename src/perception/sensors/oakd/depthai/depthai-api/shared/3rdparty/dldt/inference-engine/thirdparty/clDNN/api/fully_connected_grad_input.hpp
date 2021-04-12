/*
// Copyright (c) 2018 Intel Corporation
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
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs backward fully connected layer (inner product) for input.
struct fully_connected_grad_input : public primitive_base<fully_connected_grad_input> {
    CLDNN_DECLARE_PRIMITIVE(fully_connected_grad_input)

    /// @brief Constructs fully connected layer grad for input.
    /// @param id This primitive id.
    /// @param input_grad Input gradient primitive id.
    /// @param input Input primitive id.
    /// @param weights Primitive id containing weights data.
    fully_connected_grad_input(
        const primitive_id& id,
        const primitive_id& input_grad,
        const primitive_id& input,
        const primitive_id& weights,
        const padding& output_padding = padding())
        : primitive_base(id, {input_grad, input}, output_padding), weights(weights) {
    }

    /// @brief Primitive id containing weights data.
    primitive_id weights;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        return {weights};
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
