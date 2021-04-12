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

#include "depth_to_space_kernel_selector.h"
#include "depth_to_space_kernel_ref.h"

namespace kernel_selector {

depth_to_space_kernel_selector::depth_to_space_kernel_selector() { Attach<DepthToSpaceKernelRef>(); }

KernelsData depth_to_space_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::DEPTH_TO_SPACE);
}
}  // namespace kernel_selector
