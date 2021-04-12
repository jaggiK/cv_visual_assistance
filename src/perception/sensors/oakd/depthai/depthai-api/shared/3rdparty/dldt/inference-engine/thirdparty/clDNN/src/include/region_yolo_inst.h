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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/region_yolo.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
using region_yolo_node = typed_program_node<region_yolo>;

template <>
class typed_primitive_inst<region_yolo> : public typed_primitive_inst_base<region_yolo> {
    using parent = typed_primitive_inst_base<region_yolo>;

public:
    static layout calc_output_layout(region_yolo_node const& node);
    static std::string to_string(region_yolo_node const& node);

public:
    typed_primitive_inst(network_impl& network, region_yolo_node const& desc);
};

using region_yolo_inst = typed_primitive_inst<region_yolo>;

}  // namespace cldnn
