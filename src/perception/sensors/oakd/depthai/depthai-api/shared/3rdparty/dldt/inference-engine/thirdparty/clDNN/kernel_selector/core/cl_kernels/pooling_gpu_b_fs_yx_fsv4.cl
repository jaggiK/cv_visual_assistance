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


#include "include/include_all.cl"
#include "include/data_types.cl"

#define ALIGN_TO(val, multiple) (((val) + (multiple) - 1) / (multiple) * (multiple))

#define AS_TYPE(type, val) CAT(as_, type)(val)
#define IN_VEC4 MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define OUT_VEC4 MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)
#define CONVERT_OUT CAT(convert_, OUTPUT_TYPE)
#define CONVERT_OUT_VEC4 CAT(convert_, OUT_VEC4)

#if MAX_POOLING
    #define INIT_VAL CHAR_MIN
#elif AVG_POOLING
    #define INIT_VAL 0
#else
#error
#endif


inline int FUNC(apply_pooling)(int tmp, int in)
{
#if MAX_POOLING
    return max(tmp, in);
#elif AVG_POOLING
    return tmp + in;
#endif
}

KERNEL(pooling_gpu_b_fs_yx_fsv4)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint x    = (uint)get_global_id(0);
    const uint y    = (uint)get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint f    = (bf * 4) % ALIGN_TO(INPUT0_FEATURE_NUM, 4);
    const uint b    = (bf * 4) / ALIGN_TO(INPUT0_FEATURE_NUM, 4);

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;

    int result[4] = { INIT_VAL, INIT_VAL, INIT_VAL, INIT_VAL };

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elements = 0;
#endif

    const uint batch_and_feature_offset = GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, 0, 0);
    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        int input_offset_y = offset_y + j;
        bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;
        if(!zero_y)
        {
            for(uint i = 0; i < POOL_SIZE_X; i++)
            {
                int input_offset_x = offset_x + i;
                bool zero = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;
                if(!zero)
                {
                    const uint input_idx = batch_and_feature_offset + input_offset_y*IN_Y_PITCH + input_offset_x*IN_X_PITCH;

                    int int_data   = *((const __global int*)(input + input_idx));
                    IN_VEC4 ch4_data = AS_TYPE(IN_VEC4, int_data);
                    result[0] = FUNC_CALL(apply_pooling)(result[0], (int)ch4_data[0]);
                    result[1] = FUNC_CALL(apply_pooling)(result[1], (int)ch4_data[1]);
                    result[2] = FUNC_CALL(apply_pooling)(result[2], (int)ch4_data[2]);
                    result[3] = FUNC_CALL(apply_pooling)(result[3], (int)ch4_data[3]);

#ifdef DYNAMIC_KERNEL_DIVIDER
                    num_elements++;
#endif
                }
            }
        }
    }
#ifdef DYNAMIC_WITH_PADDING_KERNEL_DIVIDER
    const int hend = min(offset_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
    const int wend = min(offset_x + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
    const uint num_elements = (hend - offset_y) * (wend - offset_x);
#endif
#else // !CHECK_BOUNDRY
    uint input_idx = GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, offset_y, offset_x);

    for(uint j = 0; j < POOL_SIZE_Y; j++)
    {
        for(uint i = 0; i < POOL_SIZE_X; i++)
        {
            int int_data   = *((const __global int*)(input + input_idx));
            IN_VEC4 ch4_data = AS_TYPE(IN_VEC4, int_data);
            result[0] = FUNC_CALL(apply_pooling)(result[0], (int)ch4_data[0]);
            result[1] = FUNC_CALL(apply_pooling)(result[1], (int)ch4_data[1]);
            result[2] = FUNC_CALL(apply_pooling)(result[2], (int)ch4_data[2]);
            result[3] = FUNC_CALL(apply_pooling)(result[3], (int)ch4_data[3]);

            input_idx += IN_X_PITCH;
        }
        input_idx += (IN_Y_PITCH - POOL_SIZE_X*IN_X_PITCH);
    }

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elements = POOL_SIZE_X*POOL_SIZE_Y;
#endif
#endif

#if defined AVG_POOLING
#if ENABLE_ROUND
    int4 pool_result;
    for(uint i = 0; i < 4; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        result[i] = convert_int(round(((float)result[i] / max(num_elements, (uint)1))));
    #else
        result[i] = convert_int(round((float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X)));
    #endif
    }
#else
    float4 pool_result;
    for(uint i = 0; i < 4; i++) {
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
        pool_result[i] = (float)result[i] / max(num_elements, (uint)1);
    #else
        pool_result[i] = (float)result[i] / (int)(POOL_SIZE_Y * POOL_SIZE_X);
    #endif
    }
#endif  // ENABLE_ROUND
#else  // AVG_POOLING
    int4 pool_result;
    for (uint i = 0; i < 4; ++i) {
        pool_result[i] = result[i];
    }
#endif  // AVG_POOLING

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUT_VEC4 final_result = FINAL_NAME;
#else
    OUT_VEC4 final_result = CONVERT_OUT_VEC4(pool_result);
#endif

    for(uint op = 0; op < 4; op++)
    {
        final_result[op] = ACTIVATION(final_result[op], ACTIVATION_PARAMS);
    }

#if OUTPUT_LAYOUT_B_FS_YX_FSV4 || OUTPUT_LAYOUT_BYXF_AF32
    const uint output_pos = OUTPUT_GET_INDEX(b, f, y, x);
#if OUTPUT_FEATURE_NUM % 4 == 0
    *((__global OUT_VEC4*)(output + output_pos)) = final_result;
#else
    for (uint i = 0; i < 4; ++i) {
        if (f + i < OUTPUT_FEATURE_NUM) {
            output[output_pos + i] = final_result[i];
        }
    }
#endif
#else
    for (uint i = 0; i < 4; ++i) {
        if (OUTPUT_FEATURE_NUM % 4 == 0 || f + i < OUTPUT_FEATURE_NUM) {
            output[OUTPUT_GET_INDEX(b, f + i, y, x)] = final_result[i];
        }
    }
#endif
}

#undef ALIGN_TO
#undef AS_TYPE
#undef IN_VEC4
#undef OUT_VEC4
#undef CONVERT_OUT
#undef CONVERT_OUT_VEC4
#undef INIT_VAL
