// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class GatherTreeIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"GatherTreeIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    GatherTreeIE() = default;
    /// \param step_ids     Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH] with
    ///                     indices from per each step
    /// \param parent_idx   Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH] with
    ///                     parent beam indices
    /// \param max_seq_len  Tensor of shape [BATCH_SIZE] with maximum lengths for each
    ///                     sequence in the batch
    /// \param end_token    Tensor of shape [MAX_TIME, BATCH_SIZE, BEAM_WIDTH]
    GatherTreeIE(const Output<Node>& step_ids,
                 const Output<Node>& parent_idx,
                 const Output<Node>& max_seq_len,
                 const Output<Node>& end_token);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
};

}  // namespace op
}  // namespace ngraph
