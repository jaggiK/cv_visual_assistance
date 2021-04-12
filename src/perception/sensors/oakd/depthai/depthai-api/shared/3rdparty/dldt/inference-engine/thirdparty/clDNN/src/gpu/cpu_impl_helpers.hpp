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

#pragma once

#include <algorithm>
#include <vector>

namespace cldnn {
namespace cpu {

struct bounding_box {
    float xmin;
    float ymin;
    float xmax;
    float ymax;

    struct center_point_construct_tag {};
    struct two_corners_construct_tag {};
    struct corner_size_construct_tag {};

    bounding_box(float xmin, float ymin, float xmax, float ymax)
        : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}

    bounding_box() : bounding_box(0, 0, 0, 0) {}

    bounding_box(float centerx, float centery, float width, float height, center_point_construct_tag)
        : bounding_box(centerx - width / 2, centery - height / 2, centerx + width / 2, centery + width / 2) {}

    bounding_box(float ax, float ay, float bx, float by, two_corners_construct_tag)
        : bounding_box(std::min(ax, bx), std::min(ay, by), std::max(ax, bx), std::max(ay, by)) {}

    bounding_box(float xmin, float ymin, float width, float height, corner_size_construct_tag)
        : bounding_box(xmin, ymin, xmin + width, ymin + height) {}

    // Computes the area of a bounding box.
    float area() const { return (xmax - xmin) * (ymax - ymin); }
};

inline float iou(const bounding_box& box1, const bounding_box& box2) {
    float area1 = box1.area();
    float area2 = box2.area();

    float inter_xmin = std::max(box1.xmin, box2.xmin);
    float inter_xmax = std::min(box1.xmax, box2.xmax);
    float inter_ymin = std::max(box1.ymin, box2.ymin);
    float inter_ymax = std::min(box1.ymax, box2.ymax);

    float intersection_x = (inter_xmax - inter_xmin);
    float intersection_y = (inter_ymax - inter_ymin);

    if (intersection_x <= 0 || intersection_y <= 0)
        return 0.f;

    float intersection = intersection_x * intersection_y;
    float union_ = (area1 + area2 - intersection);

    if (union_ <= 0)
        return 0.f;

    return intersection / union_;
}

template <typename T>
using vector1D = std::vector<T>;

template <typename T>
using vector2D = vector1D<vector1D<T>>;

template <typename T>
using vector3D = vector2D<vector1D<T>>;

template <typename T>
using vector4D = vector2D<vector2D<T>>;

}  // namespace cpu
}  // namespace cldnn
