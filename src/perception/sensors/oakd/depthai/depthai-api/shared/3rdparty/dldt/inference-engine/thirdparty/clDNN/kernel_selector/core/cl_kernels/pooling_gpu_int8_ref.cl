// Copyright (c) 2016-2017 Intel Corporation
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

KERNEL(pooling_gpu_int8_ref)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
#if OUTPUT_LAYOUT_BFYX  || OUTPUT_LAYOUT_BYXF || OUTPUT_LAYOUT_BYXF_AF32 || OUTPUT_LAYOUT_B_FS_YX_FSV4 || OUTPUT_LAYOUT_BFZYX
    const uint x    = (uint)get_global_id(0);
    const uint yz   = (uint)get_global_id(1);
#if OUTPUT_DIMS == 5
    const uint y   = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
    const uint z   = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
#else
    const uint y   = (uint)get_global_id(1);
    const uint z   = 0;
#endif
    const uint bf   = (uint)get_global_id(2);
    const uint f    = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;

    if (x >= OUTPUT_SIZE_X)
    {
        return;
    }
#elif OUTPUT_LAYOUT_B_FS_YX_FSV32 || OUTPUT_LAYOUT_B_FS_ZYX_FSV32
    const uint fsv = get_global_id(0);
    const uint zyx = get_global_id(1);
    const uint fsb = get_global_id(2);

    const uint x = zyx % OUTPUT_SIZE_X;
#if OUTPUT_DIMS == 5
    const uint y = zyx / OUTPUT_SIZE_X % OUTPUT_SIZE_Y;
    const uint z = zyx / OUTPUT_SIZE_X / OUTPUT_SIZE_Y;
#else
    const uint y = zyx / OUTPUT_SIZE_X;
    const uint z = 0;
#endif
    const uint fs = fsb % ((OUTPUT_FEATURE_NUM + 32 - 1) / 32);
    const uint b = fsb / ((OUTPUT_FEATURE_NUM + 32 - 1) / 32);
    const uint f = fs * 32 + fsv;

    if (f >= OUTPUT_FEATURE_NUM) {
        return;
    }
#elif OUTPUT_LAYOUT_YXFB
    const uint x    = (uint)get_global_id(1);
    const uint y    = (uint)get_global_id(2);
    const uint bf   = (uint)get_global_id(0);
    const uint f    = bf / INPUT0_BATCH_NUM;
    const uint b    = bf % INPUT0_BATCH_NUM;
#else
#error "pooling_int8_ref: unsupported layout"
#endif

    const int offset_x = (int)x*STRIDE_SIZE_X - PADDING_SIZE_X;
    const int offset_y = (int)y*STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int offset_z = (int)z*STRIDE_SIZE_Z - PADDING_SIZE_Z;

    int result = INIT_VAL;

#ifdef CHECK_BOUNDRY
    if (offset_x + POOL_SIZE_X < 0 || offset_x >= INPUT0_SIZE_X ||
        offset_y + POOL_SIZE_Y < 0 || offset_y >= INPUT0_SIZE_Y ||
        offset_z + POOL_SIZE_Z < 0 || offset_z >= INPUT0_SIZE_Z)
    {
        return;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

#if OUTPUT_DIMS == 5
    for(uint l = 0; l < POOL_SIZE_Z; l++)
    {
        int input_offset_z = offset_z + l;
        bool zero_z = input_offset_z >= INPUT0_SIZE_Z || input_offset_z < 0;
        if (!zero_z)
#endif
        {
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
#if OUTPUT_DIMS == 5
                            const uint input_idx = INPUT0_GET_INDEX(b, f, input_offset_z, input_offset_y, input_offset_x);
#else
                            const uint input_idx = INPUT0_GET_INDEX(b, f, input_offset_y, input_offset_x);
#endif

                            result = FUNC_CALL(apply_pooling)(result, (int)input[input_idx]);

#ifdef DYNAMIC_KERNEL_DIVIDER
                            num_elementes++;
#endif
                        }
                    }
                }
            }
        }
#if OUTPUT_DIMS == 5
    }
#endif

#ifdef DYNAMIC_WITH_PADDING_KERNEL_DIVIDER
    const int hend = min(offset_y + POOL_SIZE_Y, INPUT0_SIZE_Y + PADDING_SIZE_Y);
    const int wend = min(offset_x + POOL_SIZE_X, INPUT0_SIZE_X + PADDING_SIZE_X);
#if OUTPUT_DIMS == 5
    const int zend = min(offset_z + POOL_SIZE_Z, INPUT0_SIZE_Z + PADDING_SIZE_Z);
    const uint num_elementes = (hend - offset_y) * (wend - offset_x) * (zend - offset_z);
#else
    const uint num_elementes = (hend - offset_y) * (wend - offset_x);
#endif

#endif  // DYNAMIC_WITH_PADDING_KERNEL_DIVIDER

#else  // CHECK_BOUNDRY

#if OUTPUT_DIMS == 5
    for(uint l = 0; l < POOL_SIZE_Z; l++)
#endif
    {
        for(uint j = 0; j < POOL_SIZE_Y; j++)
        {
            for(uint i = 0; i < POOL_SIZE_X; i++)
            {
#if OUTPUT_DIMS == 5
                uint input_idx = INPUT0_GET_INDEX(b, f, offset_z + l, offset_y + j, offset_x + i);
#else
                uint input_idx = INPUT0_GET_INDEX(b, f, offset_y + j, offset_x + i);
#endif
                result = FUNC_CALL(apply_pooling)(result, (int)input[input_idx]);
            }
        }
    }

#if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    const uint num_elementes = POOL_SIZE_X*POOL_SIZE_Y*POOL_SIZE_Z;
#endif

#endif // CHECK_BOUNDRY

#if defined AVG_POOLING
#if ENABLE_ROUND
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    int pool_res = convert_int(round((float)result / max(num_elementes, (uint)1)));
    #else
    int pool_res = convert_int(round((float)result / (int)(POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X)));
    #endif
#else  // ENABLE_ROUND
    #if defined(DYNAMIC_KERNEL_DIVIDER) || defined(DYNAMIC_WITH_PADDING_KERNEL_DIVIDER)
    float pool_res = (float)result / max(num_elementes, (uint)1);
    #else
    float pool_res = (float)result / (int)(POOL_SIZE_Z * POOL_SIZE_Y * POOL_SIZE_X);
    #endif
#endif  // ENABLE_ROUND
#else  // defined AVG_POOLING
    int pool_res = result;
#endif  // defined AVG_POOLING

#if HAS_FUSED_OPS
      FUSED_OPS;
      OUTPUT_TYPE dst = FINAL_NAME;
#else  // HAS_FUSED_OPS
      OUTPUT_TYPE dst = TO_OUTPUT_TYPE(pool_res);
#endif  // HAS_FUSED_OPS

#if OUTPUT_DIMS == 5
    const uint output_pos = OUTPUT_GET_INDEX(b, f, z, y, x);
#else
    const uint output_pos = OUTPUT_GET_INDEX(b, f, y, x);
#endif
    output[output_pos] = ACTIVATION(dst, ACTIVATION_PARAMS);
}

#undef INIT_VAL
