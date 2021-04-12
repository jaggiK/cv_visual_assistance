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

#include "pooling_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"
#include "pooling/pooling_kernel_selector.h"
#include "pooling/pooling_kernel_base.h"
#include <algorithm>

namespace cldnn {
namespace gpu {

namespace {
void validate_args(const pooling_node& arg) {
    auto const& input_buffer_size = arg.input().get_output_layout().get_buffer_size();
    auto const& input_dimensions =
        input_buffer_size.batch.size() + input_buffer_size.feature.size() + input_buffer_size.spatial.size();
    auto const& output_buffer_size = arg.get_output_layout().get_buffer_size();
    auto const& output_dimensions =
        output_buffer_size.batch.size() + output_buffer_size.feature.size() + output_buffer_size.spatial.size();
    auto& stride = arg.get_primitive()->stride;
    auto const& stride_dimensions = stride.batch.size() + stride.feature.size() + stride.spatial.size();
    auto& window = arg.get_primitive()->size;
    auto const& window_dimensions = window.batch.size() + window.feature.size() + window.spatial.size();

    CLDNN_ERROR_NOT_EQUAL(arg.id(), "input dimensions", input_dimensions, "output dimensions", output_dimensions, "");
    CLDNN_ERROR_NOT_EQUAL(arg.id(), "stride dimensions", stride_dimensions, "output dimensions", output_dimensions, "");
    CLDNN_ERROR_NOT_EQUAL(arg.id(), "window dimensions", window_dimensions, "output dimensions", output_dimensions, "");
}

kernel_selector::pool_type cldnn_2_pool_type(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
            return kernel_selector::pool_type::MAX;
        case pooling_mode::average:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::average_no_padding:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::max_with_argmax:
            return kernel_selector::pool_type::MAX_WITH_ARGMAX;
        default:
            assert(0);
            return kernel_selector::pool_type::MAX;
    }
}

kernel_selector::kernel_divider_mode cldnn_2_kernel_divider_mode(pooling_mode mode) {
    switch (mode) {
        case pooling_mode::max:
        case pooling_mode::max_with_argmax:
            return kernel_selector::kernel_divider_mode::DONT_CARE;
        case pooling_mode::average:
            return kernel_selector::kernel_divider_mode::FIXED;
        case pooling_mode::average_no_padding:
            return kernel_selector::kernel_divider_mode::DYNAMIC;
        default:
            assert(0);
            return kernel_selector::kernel_divider_mode::DONT_CARE;
    }
}
}  // namespace

struct pooling_gpu : typed_primitive_gpu_impl<pooling> {
    using parent = typed_primitive_gpu_impl<pooling>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<pooling>& instance,
                                                        int32_t split) const override {
        kernel::kernel_arguments_data args = parent::get_arguments(instance, split);
        if (!instance.argument.argmax.empty())
            args.inputs.push_back((memory_impl::cptr) &instance.dep_memory(1));
        return args;
    }

public:
    static primitive_impl* create(const pooling_node& arg) {
        validate_args(arg);

        auto pool_params = get_default_params<kernel_selector::pooling_params>(arg);
        auto pool_optional_params =
            get_default_optional_params<kernel_selector::pooling_optional_params>(arg.get_program());

        const auto primitive = arg.get_primitive();
        const auto& stride = primitive->stride;
        const auto& input_offset = primitive->input_offset;
        const auto& input_sizes = arg.input().get_output_layout().size;
        const auto& output_sizes = arg.get_output_layout().size;

        auto& pp = pool_params;

        pp.poolType = cldnn_2_pool_type(primitive->mode);
        pp.remainderAction = kernel_selector::pool_remainder::CEIL;

        if (primitive->global_pooling) {
            primitive->size.spatial[0] = input_sizes.spatial[0];
            primitive->size.spatial[1] = input_sizes.spatial[1];
            primitive->size.spatial[2] = input_sizes.spatial[2];
        }

        // check if last pooling window goes outside of input size + padding. If so the avg pooling size will be
        // adjusted to that.
        auto dynamic_mode = (((output_sizes.spatial[0] - 1) * stride.spatial[0]) + primitive->size.spatial[0]) >
                                -2 * input_offset.spatial[0] + input_sizes.spatial[0] ||
                            (((output_sizes.spatial[1] - 1) * stride.spatial[1]) + primitive->size.spatial[1]) >
                                -2 * input_offset.spatial[1] + input_sizes.spatial[1] ||
                            (((output_sizes.spatial[2] - 1) * stride.spatial[2]) + primitive->size.spatial[2]) >
                                -2 * input_offset.spatial[2] + input_sizes.spatial[2];

        if (primitive->mode == pooling_mode::average && dynamic_mode)
            pp.divMode = kernel_selector::kernel_divider_mode::DYNAMIC_WITH_PADDING;
        else
            pp.divMode = cldnn_2_kernel_divider_mode(primitive->mode);

        const auto additional_offset = tensor::max(input_offset, (tensor) 0);
        if (additional_offset != (tensor) 0) {
            const auto& input_layout = arg.input().get_output_layout();
            pool_params.inputs[0] = convert_data_tensor(input_layout, 1, additional_offset);
        }

        if (primitive->mode == pooling_mode::max_with_argmax)
            pool_params.inputs.push_back(convert_data_tensor(arg.argmax().get_output_layout()));

        pp.poolSize = {
            (uint32_t)primitive->size.spatial[0],
            (uint32_t)primitive->size.spatial[1],
            (uint32_t)primitive->size.spatial[2],
        };

        pp.poolPad = {(uint32_t)std::max(-input_offset.spatial[0], 0),
                      (uint32_t)std::max(-input_offset.spatial[1], 0),
                      (uint32_t)std::max(-input_offset.spatial[2], 0)};

        pp.poolStride = {(uint32_t)stride.spatial[0], (uint32_t)stride.spatial[1], (uint32_t)stride.spatial[2]};

        auto& kernel_selector = kernel_selector::pooling_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(pool_params, pool_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto pool = new pooling_gpu(arg, best_kernels[0]);

        return pool;
    }
};

namespace detail {

attach_pooling_gpu::attach_pooling_gpu() {
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::yxfb), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::yxfb), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::byxf), pooling_gpu::create);
    // block fp16 format
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx_f16), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx_f16), pooling_gpu::create);
    // 3D
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), pooling_gpu::create);

    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx_f16), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx_f16), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx_f16), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx_b16f16), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx_b16f16), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx_b16f16), pooling_gpu::create);
    // MMAD
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::byxf_af32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::byxf_af32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::fs_bs_yx_bsv4_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv4), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv4), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_zyx_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_zyx_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_zyx_fsv32), pooling_gpu::create);
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_zyx_fsv32), pooling_gpu::create);
    //
    implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::fs_b_yx_fsv32), pooling_gpu::create);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
