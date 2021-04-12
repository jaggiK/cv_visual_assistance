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
#include "primitive.hpp"
#include "memory.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief reorder mean operation modes
enum class reorder_mean_mode {
    none,      // val
    subtract,  // val - mean
    mul,       // val * mean
    div,       // val/mean
};

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
/// Also merged with subtraction layer, which can subtract, multiply or divide values based on mean_mode value, while doing reordering.
/// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
struct reorder : public primitive_base<reorder> {
    CLDNN_DECLARE_PRIMITIVE(reorder)

    /// @brief Constructs reorder primitive with directly provided mean subtract values.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param values_to_subtract Array of mean subtract values.
    reorder(const primitive_id& id,
            const primitive_id& input,
            const layout& output_layout,
            const std::vector<float>& values_to_subtract = {},
            const reorder_mean_mode mode = reorder_mean_mode::subtract)
        : primitive_base(id, {input}, output_layout.data_padding, optional_data_type {output_layout.data_type}),
          output_format(output_layout.format),
          mean(""),
          subtract_per_feature(values_to_subtract),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    reorder(const primitive_id& id,
            const primitive_id& input,
            const layout& output_layout,
            primitive_id const& mean,
            const reorder_mean_mode mode = reorder_mean_mode::subtract)
        : primitive_base(id, {input}, output_layout.data_padding, optional_data_type {output_layout.data_type}),
          output_format(output_layout.format),
          mean(mean),
          subtract_per_feature(0),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive with directly provided mean subtract values.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param values_to_subtract Array of mean subtract values.
    reorder(const primitive_id& id,
            const primitive_id& input,
            format output_format,
            data_types output_data_type,
            const std::vector<float>& values_to_subtract = {},
            const reorder_mean_mode mode = reorder_mean_mode::subtract,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding, optional_data_type{output_data_type}),
          output_format(output_format),
          mean(""),
          subtract_per_feature(values_to_subtract),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    reorder(const primitive_id& id,
            const primitive_id& input,
            format output_format,
            data_types output_data_type,
            primitive_id const& mean,
            const reorder_mean_mode mode = reorder_mean_mode::subtract,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding, optional_data_type {output_data_type}),
          output_format(output_format),
          mean(mean),
          subtract_per_feature(0),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input input primitive id.
    /// @param input input2 primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    reorder(const primitive_id& id,
        const primitive_id& input,
        const primitive_id& input2,
        const layout& output_layout,
        const std::vector<float>& values_to_subtract = {},
            const reorder_mean_mode mode = reorder_mean_mode::subtract)
        : primitive_base(id, { input, input2 }, output_layout.data_padding, optional_data_type { output_layout.data_type }),
          output_format(output_layout.format),
          mean(""),
          subtract_per_feature(values_to_subtract),
          mean_mode(mode) {}

    /// @brief Requested memory format.
    format output_format;
    /// @brief Primitive id to get mean subtract values. Ignored if subtract_per_featrue is set.
    primitive_id mean;
    /// @brief Array of mean subtract values.
    std::vector<float> subtract_per_feature;
    /// @brief Mode of mean execution
    reorder_mean_mode mean_mode;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        if (mean.empty())
            return {};
        return {mean};
    }
};

/// @}
/// @}
/// @}
}  // namespace cldnn
