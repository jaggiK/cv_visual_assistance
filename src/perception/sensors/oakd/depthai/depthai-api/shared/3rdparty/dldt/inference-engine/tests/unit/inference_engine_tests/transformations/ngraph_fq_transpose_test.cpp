// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "tests_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/fused/fake_quantize.hpp>
#include <transform/transformations/pull_transpose_through_fq.hpp>
#include <ngraph/pass/constant_folding.hpp>

using namespace testing;

class FQTransposeTests : public TestsCommon {};

TEST_F(FQTransposeTests, FQTransposeTest1) {
    auto data1 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 3}, {1, 2, 3});
    auto data2 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {1, 2, 3});
    auto data3 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3}, {1, 2, 3});
    auto data4 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3}, {1, 2, 3});
    auto data5 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3}, {1, 2, 3});
    auto transpose_order = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});

    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto fq = std::make_shared<ngraph::op::FakeQuantize>(data1, data2, data3, data4, data5, 1);
        auto transpose = std::make_shared<ngraph::op::Transpose>(fq, transpose_order);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose}, ngraph::ParameterVector{});
        ngraph::pass::PullTransposeThroughFQUp().run_on_function(f);
        ngraph::pass::ConstantFolding().run_on_function(f);
    }
    std::vector<size_t> ref_shape{1, 3, 1};
    for (auto op : f->get_ops()) {
        if (auto constant = ngraph::as_type_ptr<ngraph::op::Constant>(op)) {
            auto shape = constant->get_shape();
            ASSERT_EQ(shape, ref_shape);
        }
    }
}
