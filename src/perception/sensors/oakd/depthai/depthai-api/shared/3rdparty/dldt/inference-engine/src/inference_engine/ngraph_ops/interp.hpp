// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <string>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
typedef struct {
    int height = -1;
    int width = -1;
    float zoom_factor = 0;
    float shrink_factor = 0;
    float scale_factor = 1.0;
    bool align_corners = true;
    bool antialias = true;
    std::string mode = "";
    int pad_beg = 0;
    int pad_end = 0;
} InterpolateIEAttrs;

class Interp : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Interp", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Interp(const Output<Node>& image, const InterpolateIEAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    InterpolateIEAttrs get_attrs() { return m_attrs; }

private:
    InterpolateIEAttrs m_attrs;
};

typedef struct {
    bool antialias = true;
    int64_t factor = 0;
    std::string mode = "";
} ResampleIEAttrs;

class ResampleV2 : public Op {
public:
    static constexpr NodeTypeInfo type_info{"ResampleV2", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    ResampleV2(const Output<Node>& image,
               const Output<Node>& output_shape,
               const ResampleIEAttrs& attrs);

    ResampleV2(const Output<Node>& image,
               const ResampleIEAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    ResampleIEAttrs get_attrs() { return m_attrs; }
private:
    ResampleIEAttrs m_attrs;
};
}  // namespace op
}  // namespace ngraph