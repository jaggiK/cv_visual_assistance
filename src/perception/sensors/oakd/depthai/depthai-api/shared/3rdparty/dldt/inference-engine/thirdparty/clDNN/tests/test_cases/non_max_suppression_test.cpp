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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>

#include "test_utils.h"

#include "api/topology.hpp"
#include "api/network.hpp"
#include "api/input_layout.hpp"
#include "api/non_max_suppression.hpp"
#include "api/data.hpp"

using namespace cldnn;

template <typename T>
struct non_max_suppression_basic : public testing::Test {
    // Boxes:
    // batch 0:
    // 0. (0, 0) - (10, 10)
    // 1. (0, 2) - (9, 10) - iou 0: 0.72
    // 2. (5, 1) - (10, 10) - iou 0: 0.45, iou 1: 0.38
    // batch 1:
    // 0. (5, 0) - (10, 5)
    // 1. (0, 0) - (5, 5) - iou 0: 0
    // 2. (2, 0) - (9, 4) - iou 0: 0.43 iou 1: 0.29
    //
    // Scores:
    // batch.box  0.0    0.1    0.2    1.0    1.1    1.2
    // class
    // 0          0.3     0.7   0.9    0.25   0.5    0.8
    // 1          0.9     0.2   0.75   0.1    0.2    0.3
    //
    //
    // Sorted output:
    // batch  class  box  score
    //   0      0     2     0.9
    //   0      1     0     0.9
    //   1      0     2     0.8
    //   0      1     2     0.75   -- iou 0.45
    //   0      0     1     0.7    -- iou 0.38
    //   1      0     1     0.5    -- iou 0.29
    //   0      0     0     0.3    -- iou 0.72
    //   1      1     2     0.3
    //   1      0     0     0.25   -- iou 0.43
    //   0      1     1     0.2    -- iou 0.72
    //   1      1     1     0.2    -- iou 0.29
    //   1      1     0     0.1    -- iou 0.43
    const int batch_size = 2;
    const int classes_num = 2;
    const int boxes_num = 3;

    const std::vector<T> boxes_data = {
        T(0.f), T(0.f), T(10.f), T(10.f),
        T(0.f), T(2.f), T(9.f), T(10.f),
        T(5.f), T(1.f), T(10.f), T(10.f),

        T(5.f), T(0.f), T(10.f), T(5.f),
        T(0.f), T(0.f), T(5.f), T(5.f),
        T(2.f), T(0.f), T(9.f), T(4.f),
    };

    const std::vector<T> scores_data = {
        T(0.3f), T(0.7f), T(0.9f),
        T(0.9f), T(0.2f), T(0.75f),
        T(0.25f), T(0.5f), T(0.8f),
        T(0.1f), T(0.2f), T(0.3f),
    };

    const layout boxes_layout = layout(type_to_data_type<T>::value, format::bfyx, tensor(batch(batch_size), feature(boxes_num), spatial(1, 4)));
    const layout scores_layout = layout(type_to_data_type<T>::value, format::bfyx, tensor(batch(batch_size), feature(classes_num), spatial(1, boxes_num)));

    memory get_boxes_memory(engine& engine) {
        auto mem = memory::allocate(engine, boxes_layout);
        tests::set_values(mem, boxes_data);
        return mem;
    }

    memory get_scores_memory(engine& engine) {
        auto mem = memory::allocate(engine, scores_layout);
        tests::set_values(mem, scores_data);
        return mem;
    }

    const int pad = -1;
};

using nms_types = testing::Types<float, half_t>;
TYPED_TEST_CASE(non_max_suppression_basic, nms_types);

TYPED_TEST(non_max_suppression_basic, basic) {
    auto engine = tests::get_test_engine();

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(non_max_suppression("nms", "boxes", "scores", 6, false));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        0, 1, 2,
        0, 0, 1,
        1, 0, 1,
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, basic_all) {
    auto engine = tests::get_test_engine();

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(non_max_suppression("nms", "boxes", "scores", 12, false));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        0, 1, 2,
        0, 0, 1,
        1, 0, 1,
        0, 0, 0,
        1, 1, 2,
        1, 0, 0,
        0, 1, 1,
        1, 1, 1,
        1, 1, 0
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, num_per_class) {
    auto engine = tests::get_test_engine();

    auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, { 1.f });

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(non_max_suppression("nms", "boxes", "scores", 6, false, "num_per_class"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        1, 1, 2,
        this->pad, this->pad, this->pad,
        this->pad, this->pad, this->pad,
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, iou_threshold) {
    auto engine = tests::get_test_engine();

    auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, { 3.f });
    auto iou_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, { 0.4f });

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(non_max_suppression("nms", "boxes", "scores", 6, false, "num_per_class", "iou_threshold"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        0, 0, 1,
        1, 0, 1,
        1, 1, 2
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}

TYPED_TEST(non_max_suppression_basic, score_threshold) {
    auto engine = tests::get_test_engine();

    auto num_per_class_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(num_per_class_mem, { 3.f });
    auto iou_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(iou_threshold_mem, { 0.4f });
    auto score_threshold_mem = memory::allocate(engine, layout(data_types::f32, format::bfyx, tensor(batch(1))));
    tests::set_values(score_threshold_mem, { 0.4f });

    topology topo;
    topo.add(input_layout("boxes", this->boxes_layout));
    topo.add(input_layout("scores", this->scores_layout));
    topo.add(data("num_per_class", num_per_class_mem));
    topo.add(data("iou_threshold", iou_threshold_mem));
    topo.add(data("score_threshold", score_threshold_mem));
    topo.add(non_max_suppression("nms", "boxes", "scores", 6, false, "num_per_class", "iou_threshold", "score_threshold"));

    build_options build_opts(
        build_option::optimize_data(true)
    );
    auto net = network(engine, topo, build_opts);

    auto boxes_mem = this->get_boxes_memory(engine);
    auto scores_mem = this->get_scores_memory(engine);

    net.set_input_data("boxes", boxes_mem);
    net.set_input_data("scores", scores_mem);

    auto result = net.execute();

    std::vector<int> expected_out = {
        0, 0, 2,
        0, 1, 0,
        1, 0, 2,
        0, 0, 1,
        1, 0, 1,
        this->pad, this->pad, this->pad,
    };

    auto out_mem = result.at("nms").get_memory();
    auto out_ptr = out_mem.pointer<int>();

    ASSERT_EQ(expected_out.size(), out_ptr.size());
    for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_EQ(expected_out[i], out_ptr[i]) << "at i = " << i;
    }
}
