// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"

TEST_F(NGraphReaderTests, ReadFQNetwork) {
    std::string model = R"V0G0N(
<net name="FakeQuantize" version="10">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,56,96,168"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>96</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const_1" precision="FP32" type="Const" version="opset1">
            <data offset="14272" size="224"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="const_2" precision="FP32" type="Const" version="opset1">
            <data offset="14272" size="224"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="const_3" precision="FP32" type="Const" version="opset1">
            <data offset="14496" size="4"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="const_4" precision="FP32" type="Const" version="opset1">
            <data offset="14500" size="4"/>
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="FakeQuantize" precision="FP32" type="FakeQuantize" version="opset1">
            <data levels="2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>96</dim>
                    <dim>168</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>96</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>96</dim>
                    <dim>168</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="FakeQuantize" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="in1" type="Input" precision="FP32">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>96</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="const_1" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="14272" size="224"/>
            </blobs>
        </layer>
        <layer id="2" name="const_2" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="14272" size="224"/>
            </blobs>
        </layer>
        <layer id="3" name="const_3" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="14496" size="4"/>
            </blobs>
        </layer>
        <layer id="4" name="const_4" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="14500" size="4"/>
            </blobs>
        </layer>
        <layer id="5" name="FakeQuantize" precision="FP32" type="FakeQuantize">
            <data levels="2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>96</dim>
                    <dim>168</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>56</dim>
                    <dim>96</dim>
                    <dim>168</dim>
                </port>
            </output>
        </layer>

    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
    </edges>
</net>
)V0G0N";
    IRReader reader;

    Blob::Ptr weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {139392}, Layout::C));
    weights->allocate();
    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    auto nGraph = reader.read(model, weights);
    CNNNetwork cnn(nGraph);
    cnn.begin();
}
