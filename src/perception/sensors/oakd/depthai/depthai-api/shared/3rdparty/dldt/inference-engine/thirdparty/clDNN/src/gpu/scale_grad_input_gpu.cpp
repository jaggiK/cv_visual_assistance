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

#include "scale_grad_input_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct scale_grad_input_gpu : typed_primitive_gpu_impl<scale_grad_input> {
    using parent = typed_primitive_gpu_impl<scale_grad_input>;
    using parent::parent;

protected:
    kernel::kernel_arguments_data get_arguments(typed_primitive_inst<scale_grad_input>& instance,
                                                        int32_t) const override {
        kernel::kernel_arguments_data args;
        args.inputs = { (memory_impl::cptr) &instance.input_memory(), (memory_impl::cptr) &instance.scale_input_memory()};
        args.output = (memory_impl::cptr) &instance.output_memory();

        return args;
    }

public:
    static primitive_impl* create(const scale_grad_input_node& arg) {
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params =
            get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.inputs.push_back(convert_data_tensor(arg.scale_in().get_output_layout()));

        ew_params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0),
                                         kernel_selector::eltwise_params::InputType::Buffer(1)},
                                        kernel_selector::eltwise_mode::MUL});

        ew_params.layoutBased = true;

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto scale_grad_input = new scale_grad_input_gpu(arg, best_kernels[0]);

        return scale_grad_input;
    }
};

namespace detail {

attach_scale_grad_input_gpu::attach_scale_grad_input_gpu() {
    auto val_fw = scale_grad_input_gpu::create;

    implementation_map<scale_grad_input>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb),
                                              val_fw);
    implementation_map<scale_grad_input>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb),
                                              val_fw);
    implementation_map<scale_grad_input>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                              val_fw);
    implementation_map<scale_grad_input>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                              val_fw);
    implementation_map<scale_grad_input>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::byxf),
                                              val_fw);
    implementation_map<scale_grad_input>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::byxf),
                                              val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
