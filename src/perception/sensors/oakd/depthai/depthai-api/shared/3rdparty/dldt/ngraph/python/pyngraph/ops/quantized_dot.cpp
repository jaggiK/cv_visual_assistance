//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/op/quantized_dot.hpp"
#include "pyngraph/ops/quantized_dot.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_QuantizedDot(py::module m)
{
    py::class_<ngraph::op::QuantizedDot, std::shared_ptr<ngraph::op::QuantizedDot>, ngraph::op::Op>
        quantizeddot(m, "QuantizedDot");
    quantizeddot.doc() = "ngraph.impl.op.QuantizedDot wraps ngraph::op::QuantizedDot";
    quantizeddot.def(py::init<const std::shared_ptr<ngraph::Node>&,
                              const std::shared_ptr<ngraph::Node>&,
                              const int,
                              const std::shared_ptr<ngraph::Node>&,
                              const std::shared_ptr<ngraph::Node>&,
                              const std::shared_ptr<ngraph::Node>&,
                              const std::shared_ptr<ngraph::Node>&,
                              const std::shared_ptr<ngraph::Node>&,
                              const std::shared_ptr<ngraph::Node>&,
                              const ngraph::element::Type&,
                              const ngraph::AxisSet&,
                              const ngraph::AxisSet&,
                              const ngraph::AxisSet&>());
}
