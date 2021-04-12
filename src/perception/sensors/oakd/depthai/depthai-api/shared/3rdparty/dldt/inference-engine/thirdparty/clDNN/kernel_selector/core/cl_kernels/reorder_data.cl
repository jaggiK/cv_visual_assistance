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


#include "include/reshape_dims.cl"
#include "include/fetch.cl"

#include "include/data_types.cl"

///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 6
    return GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#else
#error reorder_data.cl: input format - not supported
#endif
}

inline uint FUNC(get_input3d_index)(uint b, uint f, uint z, uint y, uint x)
{
    return GET_DATA_INDEX_5D(INPUT0, b, f, z, y, x);
}
///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 6
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, y, x);
#else
#error reorder_data.cl: output format - not supported
#endif
}

KERNEL (reorder_data)(
#if defined INPUT0_LAYOUT_NV12
    read_only image2d_t input,
#else
    const __global INPUT_REORDER_TYPE* input,
#endif
    __global OUTPUT_REORDER_TYPE* output
#ifdef MEAN_SUBTRACT_IN_BUFFER
    , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
    )
{
    const uint b = get_global_id(GWS_BATCH);
    const uint f = get_global_id(GWS_FEATURE);
#if   INPUT0_DIMS == 2
    const uint y = 0;
    const uint x = 0;
    const uint z = 0;
    const uint w = 0;
#elif INPUT0_DIMS == 4
    const uint y = ((uint)(get_global_id(GWS_YX))) / INPUT0_SIZE_X;
    const uint x = ((uint)(get_global_id(GWS_YX))) % INPUT0_SIZE_X;
    const uint z = 0;
    const uint w = 0;
#elif INPUT0_DIMS == 5
    uint data_idx = get_global_id(GWS_YX);
    uint tmp_data_idx = data_idx / INPUT0_SIZE_X;
    const uint x = data_idx - tmp_data_idx * INPUT0_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / INPUT0_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * INPUT0_SIZE_Y;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / INPUT0_SIZE_Z;
    const uint z = data_idx - tmp_data_idx * INPUT0_SIZE_Z;
    const uint w = 0;
#elif INPUT0_DIMS == 6
    const uint gid_yx = (uint)(get_global_id(GWS_YX));
    const uint x = gid_yx % INPUT0_SIZE_X;
    const uint y = gid_yx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint z = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = gid_yx / INPUT0_SIZE_X / INPUT0_SIZE_Y / INPUT0_SIZE_Z % INPUT0_SIZE_W;
#endif

#if defined INPUT0_LAYOUT_NV12
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;
    float4 colorVYU = read_imagef(input, sampler, (int2)(x, y));
    float Ycomponent = colorVYU.s1 * 255;
    float Ucomponent = colorVYU.s2 * 255;
    float Vcomponent = colorVYU.s0 * 255;
    uchar R = convert_uchar_sat(1.164f * (Ycomponent - 16) + 1.596f * (Vcomponent - 128));
    uchar G = convert_uchar_sat(1.164f * (Ycomponent - 16) - 0.813f * (Vcomponent - 128) - 0.391f * (Ucomponent - 128));
    uchar B = convert_uchar_sat(1.164f * (Ycomponent - 16) + 2.018f * (Ucomponent - 128));

#else
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, f, w, z, y, x);
    const uint input_idx  = FUNC_CALL(get_input_index)(b, f, w, z, y, x);
    const uint output_idx = FUNC_CALL(get_output_index)(ov[0],ov[1],ov[2],ov[3], ov[4], ov[5]);

#if defined MEAN_SUBTRACT_INSIDE_PARAMS
    float res = TO_MEAN_TYPE(input[input_idx]);
    res = MEAN_OP(res, VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
#elif defined MEAN_SUBTRACT_IN_BUFFER
#if defined MEAN_PER_FEATURE
    MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
    res = MEAN_OP(res, mean_subtract[f]);
#else
    // TODO Add support for 6D mean
    MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
    uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, w, z, y, x);
    res = MEAN_OP(res, mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[0], msv[1], /*msv[2], msv[3],*/ msv[4], msv[5])]);
#endif
#else
    CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
#endif
#endif

#if defined INPUT0_LAYOUT_NV12
    uint8 ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 0, w, z, y, x);
    uint output_idx = FUNC_CALL(get_output_index)(ov[0], ov[1], ov[2], ov[3], ov[4], ov[5]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(R), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 1, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[0], ov[1], ov[2], ov[3], ov[4], ov[5]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(G), NL_M, NL_N);
    ov = RESHAPE_DIMS(INPUT0, OUTPUT, b, 2, w, z, y, x);
    output_idx = FUNC_CALL(get_output_index)(ov[0], ov[1], ov[2], ov[3], ov[4], ov[5]);
    output[output_idx] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(B), NL_M, NL_N);
#else
#if INPUT0_IS_FP && !OUTPUT_IS_FP
    // TODO: check if this round really needed. Right now it's added to have the same behavior as CPU plugin
    // becuase CPU's convert instruction performs round
    output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(round(res)), ACTIVATION_PARAMS_TYPED);
#else
    output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(res), ACTIVATION_PARAMS_TYPED);
#endif
#endif
}
