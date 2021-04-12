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

#include "reverse_sequence_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "reverse_sequence/reverse_sequence_kernel_selector.h"
#include "reverse_sequence/reverse_sequence_kernel_ref.h"
#include "error_handler.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct reverse_sequence_gpu : typed_primitive_gpu_impl<reverse_sequence> {
    using parent = typed_primitive_gpu_impl<reverse_sequence>;
    using parent::parent;

public:
    static primitive_impl* create(const reverse_sequence_node& arg) {
        auto reverse_sequence_params = get_default_params<kernel_selector::reverse_sequence_params>(arg);
        auto reverse_sequence_optional_params =
            get_default_optional_params<kernel_selector::reverse_sequence_optional_params>(arg.get_program());

        reverse_sequence_params.seq_axis = arg.get_primitive()->seq_axis;
        reverse_sequence_params.batch_axis = arg.get_primitive()->batch_axis;

        reverse_sequence_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::reverse_sequence_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(reverse_sequence_params, reverse_sequence_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reverse_sequence = new reverse_sequence_gpu(arg, best_kernels[0]);

        return reverse_sequence;
    }
};

namespace detail {

attach_reverse_sequence_gpu::attach_reverse_sequence_gpu() {
    auto val_fw = reverse_sequence_gpu::create;
    implementation_map<reverse_sequence>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx),
                                              val_fw);
    implementation_map<reverse_sequence>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx),
                                              val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
