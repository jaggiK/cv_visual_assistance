// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"
#include <string>

TEST_F(NGraphReaderTests, ReadMatMulNetwork1) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2048"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const" version="opset1">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>2048</dim>
                    <dim>1000</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
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
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8192000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork2) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2048"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const" version="opset1">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <data transpose_b="True" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
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
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8192000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork3) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter" version="opset1">
            <data element_type="f32" shape="2048,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="embedded_input__const" type="Const" version="opset1">
            <data offset="0" size="8192000"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <data transpose_a="True" transpose_b="True" />
            <input>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="fc/transpose_a" precision="FP32" type="Permute">
            <data order="1,0"/>
            <input>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" precision="FP32" type="FullyConnected">
            <data alpha="0" beta="0" out-size="1000"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
            <weights offset="0" size="8192000" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8192000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork4) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data1" type="Parameter" version="opset1">
            <data element_type="f32" shape="2048,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1000,2048"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" type="MatMul" version="opset1">
            <data transpose_a="True" transpose_b="True" />
            <input>
                <port id="0" precision="FP32">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="3" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="fc" precision="FP32" type="Gemm">
            <data transpose_a="True" transpose_b="True" />
            <input>
                <port id="0">
                    <dim>2048</dim>
                    <dim>1</dim>
                </port>
                <port id="1">
                    <dim>1000</dim>
                    <dim>2048</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 8192000);
}

TEST_F(NGraphReaderTests, ReadMatMulNetwork5) {
    std::string model = R"V0G0N(
<net name="Convolution" version="10">
    <layers>
        <layer id="0" name="data1" type="Parameter" version="opset1">
            <data element_type="f32" shape="2,3,2"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" type="Parameter" version="opset1">
            <data element_type="f32" shape="3,2,2,1"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" type="MatMul" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="3" from-port="3" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::string modelV5 = R"V0G0N(
<net name="Convolution" version="5" precision="FP32" batch="1">
    <layers>
        <layer id="0" name="data1" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Constant" precision="I64" type="Const">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="32"/>
            </blobs>
        </layer>
        <layer id="2" name="fc/reshape" precision="FP32" type="Reshape">
            <input>
                <port id="0">
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="fc" precision="FP32" type="Gemm">
            <data transpose_a="False" transpose_b="False" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                </port>
                <port id="1">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="4" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";
    compareIRs(model, modelV5, 48);
}
