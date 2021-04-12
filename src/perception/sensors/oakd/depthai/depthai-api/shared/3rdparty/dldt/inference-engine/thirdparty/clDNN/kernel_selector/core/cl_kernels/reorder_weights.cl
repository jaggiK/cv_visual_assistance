// Copyright (c) 2016-2019 Intel Corporation
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


#include "include/fetch.cl"
#include "include/reshape_dims.cl"
#include "include/data_types.cl"


///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint o, uint i, uint z, uint y, uint x)
{
#if   INPUT0_SIMPLE && INPUT0_DIMS <= 4
    return GET_FILTER_INDEX(INPUT0, o, i, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_FILTER_INDEX_5D(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16    || \
      defined INPUT0_LAYOUT_OS_I_OSV16      || \
      defined INPUT0_LAYOUT_OS_I_OSV8__AI8  || \
      defined INPUT0_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IYX_OSV32
    return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_IYX_OSV64
    return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, 64);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_I_YXS_OS_YXSV2_OSV16
    return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
    return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
    #error - not supported yet
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
	return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_IS_O_YX_ISV32
    return GET_FILTER_IS_O_YX_ISV32(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_IS_O32_YX_ISV32_SWIZZLED_BY_4
    return GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_Y_X8_OSV8_ISV4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_O_I_YX_I16_O16
    return GET_FILTER_O_I_YX_I16_O16_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OIYX_O16
    return GET_FILTER_OIYX_O16(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_O_I_ZYX_I16_O16
    return GET_FILTER_O_I_ZYX_I16_O16_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_I_O_ZYX_O16_I16
    return GET_FILTER_I_O_ZYX_O16_I16_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IS_OSV32_ISV32_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_O_I_ZYX_I8_O16_I2
    return GET_FILTER_O_I_ZYX_I8_O16_I2_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OZYXI_O16
    return GET_FILTER_OZYXI_O16(INPUT0, o, i, z, y, x);
#else
#error reorder_weights.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint o, uint i, uint z, uint y, uint x)
{
#if   OUTPUT_SIMPLE && OUTPUT_DIMS <= 4
    return GET_FILTER_INDEX(OUTPUT, o, i, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 5
    return GET_FILTER_INDEX_5D(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16    || \
      defined OUTPUT_LAYOUT_OS_I_OSV16      || \
      defined OUTPUT_LAYOUT_OS_I_OSV8__AI8  || \
      defined OUTPUT_LAYOUT_OS_I_OSV16__AI8
    return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV32
    return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV64
    return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, 64);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16_ROTATE_180
    return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_I_YXS_OS_YXSV2_OSV16
    return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
    return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
    return 0; //will not be used for images
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
	return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISA8_OSV8_ISV4
    return GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_IS_O_YX_ISV32
    return GET_FILTER_IS_O_YX_ISV32(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_IS_O32_YX_ISV32_SWIZZLED_BY_4
    return GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_Y_X8_OSV8_ISV4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV16_ISV4
    return GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2
    return GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_O_I_YX_I16_O16
    return GET_FILTER_O_I_YX_I16_O16_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OIYX_O16
    return GET_FILTER_OIYX_O16(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_O_I_ZYX_I16_O16
    return GET_FILTER_O_I_ZYX_I16_O16_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_I_O_ZYX_O16_I16
    return GET_FILTER_I_O_ZYX_O16_I16_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_OSV32_ISV32_SWIZZLED_BY_4
    return GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_O_I_ZYX_I8_O16_I2
    return GET_FILTER_O_I_ZYX_I8_O16_I2_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OZYXI_O16
    return GET_FILTER_OZYXI_O16(OUTPUT, o, i, z, y, x);
#else
#error reorder_weights.cl: output format - not supported
#endif
}

#if OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, write_only image2d_t output)
{
    const unsigned o = get_global_id(0);
    const unsigned iyx = get_global_id(1);
    const unsigned x = iyx % INPUT0_SIZE_X;
    const unsigned y = (iyx / INPUT0_SIZE_X) % INPUT0_SIZE_Y;
    const unsigned i = (iyx / INPUT0_SIZE_X) / INPUT0_SIZE_Y;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 4) input_val = (MAKE_VECTOR_TYPE(UNIT_TYPE, 4))(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
    const int2 coord = (int2)(o, iyx);
    uint8 ir = RESHAPE_WEIGHT_DIMS(OUTPUT, INPUT0, o, i, 0, 0, y, x);
    input_val.s0 = TO_OUTPUT_TYPE(input[FUNC_CALL(get_input_index)(ir[0],ir[1],ir[3],ir[4],ir[5])]);
    IMAGE_WRITE(output, coord, input_val);
}
#else
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const unsigned o = get_global_id(0);
    const unsigned i = get_global_id(1);
#if   OUTPUT_DIMS == 2
    const unsigned z = 0;
    const unsigned y = 0;
    const unsigned x = 0;
#elif OUTPUT_DIMS == 4
    const unsigned z = 0;
    const unsigned y = (uint)get_global_id(2) / INPUT0_SIZE_X;
    const unsigned x = (uint)get_global_id(2) % INPUT0_SIZE_X;
#elif OUTPUT_DIMS == 5
    const unsigned zyx = get_global_id(2);
    const unsigned x = zyx % INPUT0_SIZE_X;
    const unsigned y = (zyx / INPUT0_SIZE_X) % INPUT0_SIZE_Y;
    const unsigned z = (zyx / INPUT0_SIZE_X) / INPUT0_SIZE_Y;
#endif
    uint8 ir = RESHAPE_WEIGHT_DIMS(OUTPUT, INPUT0, o, i, 0, z, y, x);
    output[FUNC_CALL(get_output_index)(o, i, z, y, x)] = TO_OUTPUT_TYPE(input[FUNC_CALL(get_input_index)(ir[0],ir[1],ir[3],ir[4],ir[5])]);
}
#endif
