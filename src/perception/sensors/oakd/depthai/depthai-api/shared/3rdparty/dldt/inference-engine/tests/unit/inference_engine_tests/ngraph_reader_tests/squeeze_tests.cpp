// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"
#include <string>

TEST_F(NGraphReaderTests, ReadSqueeze) {
    std::string model = R"V0G0N(
<net name="Reshape" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2048,1,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const" version="opset1">
            <data offset="0" size="8"/>
            <output>
                <port id="1">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="squeeze" precision="FP32" type="Squeeze" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const1" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="8"/>
            </blobs>
        </layer>
        <layer id="3" name="squeeze" precision="FP32" type="Squeeze">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 24, [](Blob::Ptr& weights) {
                auto* w = weights->buffer().as<int64_t*>();
                w[0] = 3;
            });
}
