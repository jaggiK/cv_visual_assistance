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
#include "api/activation_grad.hpp"
#include "api/apply_adam.hpp"
#include "api/arg_max_min.hpp"
#include "api/average_unpooling.hpp"
#include "api/batch_norm.hpp"
#include "api/batch_norm_grad.hpp"
#include "api/binary_convolution.hpp"
#include "api/border.hpp"
#include "api/broadcast.hpp"
#include "api/concatenation.hpp"
#include "api/condition.hpp"
#include "api/contract.hpp"
#include "api/convolution.hpp"
#include "api/convolution_grad_weights.hpp"
#include "api/crop.hpp"
#include "api/custom_gpu_primitive.hpp"
#include "api/data.hpp"
#include "api/deconvolution.hpp"
#include "api/depth_to_space.hpp"
#include "api/detection_output.hpp"
#include "api/eltwise.hpp"
#include "api/embed.hpp"
#include "api/fully_connected.hpp"
#include "api/fully_connected_grad_input.hpp"
#include "api/fully_connected_grad_weights.hpp"
#include "api/gather.hpp"
#include "api/gemm.hpp"
#include "api/index_select.hpp"
#include "api/input_layout.hpp"
#include "api/lookup_table.hpp"
#include "api/lrn.hpp"
#include "api/lstm.hpp"
#include "api/lstm_dynamic.hpp"
#include "api/max_unpooling.hpp"
#include "api/mutable_data.hpp"
#include "api/mvn.hpp"
#include "api/normalize.hpp"
#include "api/one_hot.hpp"
#include "api/permute.hpp"
#include "api/pooling.hpp"
#include "api/prior_box.hpp"
#include "api/proposal.hpp"
#include "api/pyramid_roi_align.hpp"
#include "api/quantize.hpp"
#include "api/reduce.hpp"
#include "api/region_yolo.hpp"
#include "api/reorder.hpp"
#include "api/reorg_yolo.hpp"
#include "api/reshape.hpp"
#include "api/reverse_sequence.hpp"
#include "api/roi_pooling.hpp"
#include "api/scale.hpp"
#include "api/scale_grad_input.hpp"
#include "api/scale_grad_weights.hpp"
#include "api/select.hpp"
#include "api/shuffle_channels.hpp"
#include "api/softmax.hpp"
#include "api/softmax_loss_grad.hpp"
#include "api/strided_slice.hpp"
#include "api/tile.hpp"
#include "api/resample.hpp"
#include "api/gather_tree.hpp"
#include "api_extension/fused_conv_bn_scale.hpp"
#include "api_extension/fused_conv_eltwise.hpp"
#include "api_extension/lstm_dynamic_input.hpp"
#include "api_extension/lstm_dynamic_timeloop.hpp"
#include "generic_layer.hpp"
#include "api/non_max_suppression.hpp"


namespace cldnn { namespace gpu {
void register_implementations_gpu();

namespace detail {

#define REGISTER_GPU(prim)              \
    struct attach_##prim##_gpu {        \
        attach_##prim##_gpu();          \
    }

REGISTER_GPU(activation);
REGISTER_GPU(activation_grad);
REGISTER_GPU(apply_adam);
REGISTER_GPU(arg_max_min);
REGISTER_GPU(average_unpooling);
REGISTER_GPU(batch_norm);
REGISTER_GPU(batch_norm_grad);
REGISTER_GPU(binary_convolution);
REGISTER_GPU(border);
REGISTER_GPU(broadcast);
REGISTER_GPU(concatenation);
REGISTER_GPU(condition);
REGISTER_GPU(contract);
REGISTER_GPU(convolution);
REGISTER_GPU(convolution_grad_weights);
REGISTER_GPU(crop);
REGISTER_GPU(custom_gpu_primitive);
REGISTER_GPU(data);
REGISTER_GPU(deconvolution);
REGISTER_GPU(deformable_conv);
REGISTER_GPU(deformable_interp);
REGISTER_GPU(depth_to_space);
REGISTER_GPU(detection_output);
REGISTER_GPU(eltwise);
REGISTER_GPU(embed);
REGISTER_GPU(fully_connected);
REGISTER_GPU(fully_connected_grad_input);
REGISTER_GPU(fully_connected_grad_weights);
REGISTER_GPU(gather);
REGISTER_GPU(gemm);
REGISTER_GPU(index_select);
REGISTER_GPU(input_layout);
REGISTER_GPU(lookup_table);
REGISTER_GPU(lrn);
REGISTER_GPU(lstm_gemm);
REGISTER_GPU(lstm_elt);
REGISTER_GPU(max_unpooling);
REGISTER_GPU(mutable_data);
REGISTER_GPU(mvn);
REGISTER_GPU(normalize);
REGISTER_GPU(one_hot);
REGISTER_GPU(permute);
REGISTER_GPU(pooling);
REGISTER_GPU(prior_box);
REGISTER_GPU(proposal);
REGISTER_GPU(pyramid_roi_align);
REGISTER_GPU(quantize);
REGISTER_GPU(reduce);
REGISTER_GPU(region_yolo);
REGISTER_GPU(reorder);
REGISTER_GPU(reorg_yolo);
REGISTER_GPU(reshape);
REGISTER_GPU(reverse_sequence);
REGISTER_GPU(roi_pooling);
REGISTER_GPU(scale);
REGISTER_GPU(scale_grad_input);
REGISTER_GPU(scale_grad_weights);
REGISTER_GPU(select);
REGISTER_GPU(shuffle_channels);
REGISTER_GPU(softmax);
REGISTER_GPU(softmax_loss_grad);
REGISTER_GPU(strided_slice);
REGISTER_GPU(tile);
REGISTER_GPU(fused_conv_bn_scale);
REGISTER_GPU(fused_conv_eltwise);
REGISTER_GPU(lstm_dynamic_input);
REGISTER_GPU(lstm_dynamic_timeloop);
REGISTER_GPU(generic_layer);
REGISTER_GPU(gather_tree);
REGISTER_GPU(resample);
REGISTER_GPU(non_max_suppression);

#undef REGISTER_GPU

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
