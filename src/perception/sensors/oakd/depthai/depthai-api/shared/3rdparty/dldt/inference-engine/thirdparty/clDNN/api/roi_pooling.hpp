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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "pooling.hpp"
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

struct roi_pooling : public primitive_base<roi_pooling> {
    CLDNN_DECLARE_PRIMITIVE(roi_pooling)

    roi_pooling(const primitive_id& id,
                const primitive_id& input_data,
                const primitive_id& input_rois,
                pooling_mode mode,
                bool position_sensitive,
                int pooled_width,
                int pooled_height,
                float spatial_scale,
                int output_dim = 0,
                int spatial_bins_x = 1,
                int spatial_bins_y = 1,
                const padding& output_padding = padding())
        : primitive_base(id, {input_data, input_rois}, output_padding),
          mode(mode),
          position_sensitive(position_sensitive),
          pooled_width(pooled_width),
          pooled_height(pooled_height),
          spatial_scale(spatial_scale),
          trans_std(0.0f),
          no_trans(false),
          output_dim(output_dim),
          part_size(0),
          group_size(0),
          spatial_bins_x(spatial_bins_x),
          spatial_bins_y(spatial_bins_y) {}

    roi_pooling(const primitive_id& id,
                const std::vector<primitive_id>& inputs,
                pooling_mode mode,
                bool position_sensitive,
                int pooled_width,
                int pooled_height,
                float spatial_scale,
                float trans_std,
                bool no_trans,
                int part_size,
                int group_size,
                int output_dim = 0,
                int spatial_bins_x = 1,
                int spatial_bins_y = 1,
                const padding& output_padding = padding())
        : primitive_base(id, {inputs}, output_padding),
          mode(mode),
          position_sensitive(position_sensitive),
          pooled_width(pooled_width),
          pooled_height(pooled_height),
          spatial_scale(spatial_scale),
          trans_std(trans_std),
          no_trans(no_trans),
          output_dim(output_dim),
          part_size(part_size),
          group_size(group_size),
          spatial_bins_x(spatial_bins_x),
          spatial_bins_y(spatial_bins_y) {}

    pooling_mode mode;
    bool position_sensitive;
    int pooled_width;
    int pooled_height;
    float spatial_scale;
    float trans_std;
    bool no_trans;
    int output_dim;
    int part_size;
    int group_size;
    int spatial_bins_x;
    int spatial_bins_y;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
