/*
// Copyright (c) 2018 Intel Corporation
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



#include "include/include_all.cl"

#define VECTOR_TYPE MAKE_VECTOR_TYPE(UNIT_TYPE,8)
#define ACCUMULATOR_VECTOR_TYPE MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 8)
#define FEATURE_PER_ITEM 8
#define FEATURE_BLOCK_NUM (OUTPUT_FEATURE_NUM / 8)

KERNEL(lrn_within_channel_byxf_opt)(__global const INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint b = get_global_id(GWS_BATCH);
    const uint f = (uint)get_global_id(GWS_FEATURE)*FEATURE_PER_ITEM;
    const uint y = (uint)get_global_id(GWS_YX) / INPUT0_SIZE_X;
    const uint x = (uint)get_global_id(GWS_YX) % INPUT0_SIZE_X;

    const uint input_index = GET_DATA_INDEX(INPUT0, b, f, y, x);
    const uint output_index = GET_DATA_INDEX(OUTPUT, b, f, y, x);

    ACCUMULATOR_VECTOR_TYPE sum = 0.0f;
#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

    const int x_start = ((int)x - PADDING);
    const int y_start = ((int)y - PADDING);
    int input_offset = (GET_DATA_INDEX(INPUT0, b, f, y_start, x_start))/8;
    
    VECTOR_TYPE feature_block;

    for (int j = 0; j < LOCAL_SIZE; ++j)
    {
        for (int i = 0; i < LOCAL_SIZE; ++i) 
        {
            int input_offset_x = x_start + i;
            int input_offset_y = y_start + j;
            bool zero = false;
            zero = input_offset_x < 0 ? true : zero;
            zero = input_offset_y < 0 ? true : zero;
            zero = input_offset_x >= INPUT0_SIZE_X ? true : zero;
            zero = input_offset_y >= INPUT0_SIZE_Y ? true : zero;

            VECTOR_TYPE val = zero ? UNIT_VAL_ZERO : vload8(input_offset+FEATURE_BLOCK_NUM*i, input);
            
            sum = mad(val,val,sum);
#ifdef DYNAMIC_KERNEL_DIVIDER
            num_elementes += zero ? 0 : 1;
#endif
        }
        input_offset += INPUT0_Y_PITCH/FEATURE_PER_ITEM;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER 
    const UNIT_TYPE num_elementes_div = UNIT_VAL_ONE / TO_UNIT_TYPE(num_elementes);
#else
    const UNIT_TYPE num_elementes_div = NUM_ELEMENTS_DIV;
#endif
    
    const VECTOR_TYPE base = mad((ACCUMULATOR_TYPE)ALPHA*num_elementes_div, sum, TO_UNIT_TYPE(K));
    const VECTOR_TYPE normalization_factor = native_powr(base, TO_UNIT_TYPE(-BETA));
    const VECTOR_TYPE val = vload8(input_index/FEATURE_PER_ITEM, input);
    const VECTOR_TYPE normres = val*normalization_factor;

    for(uint i = 0; i < FEATURE_PER_ITEM; i++)
    {
        output[output_index+i] = ACTIVATION(normres[i], ACTIVATION_PARAMS);
    }
}

#undef FEATURE_BLOCK_NUM
#undef FEATURE_PER_ITEM
#undef VECTOR_TYPE
#undef ACCUMULATOR_VECTOR_TYPE
