/*
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
*/

#include "include/common.cl"
#include "include/data_types.cl"


#define DATA_PER_WORKITEM ( (INPUT0_CLASS_NUM + (WORKITEMS_PER_CLASSES - 1) ) / WORKITEMS_PER_CLASSES)
#define FULL_ITERATIONS_NUM (INPUT0_CLASS_NUM / WORKITEMS_PER_CLASSES)

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(softmax_items_class_optimized)(__global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if INPUT0_DIMS == 5
    const uint other0 = (uint)get_group_id(0) % INPUT0_OTHER0_SIZE;
    const uint other2 = (uint)get_group_id(0) / INPUT0_OTHER0_SIZE;
#else
    const uint other0 = get_group_id(0);
    const uint other2 = 0;
#endif
    const uint other1 = get_group_id(1);
    const uint batch  = get_group_id(2);
    const uint simd_lane = get_sub_group_local_id();

    const uint in_depth_offset  = batch*INPUT0_BATCH_PITCH + other2*INPUT0_OTHER2_PITCH + other1*INPUT0_OTHER1_PITCH + other0*INPUT0_OTHER0_PITCH + INPUT0_OFFSET;
    const uint out_depth_offset = batch*OUTPUT_BATCH_PITCH + other2*OUTPUT_OTHER2_PITCH + other1*OUTPUT_OTHER1_PITCH + other0*OUTPUT_OTHER0_PITCH + OUTPUT_OFFSET;

    UNIT_TYPE max_value = UNIT_VAL_MIN;
    UNIT_TYPE data[DATA_PER_WORKITEM];

    // PART 1. Calculate MAX value
    uint input_idx = in_depth_offset + simd_lane * INPUT0_CLASS_PITCH;
    for (uint cls = 0; cls < FULL_ITERATIONS_NUM; cls++)
    {
        UNIT_TYPE in = input[input_idx];
        max_value = max(max_value, in);
        data[cls] = in;
        input_idx += WORKITEMS_PER_CLASSES*INPUT0_CLASS_PITCH;
    }
    if(simd_lane < LEFTOVERS)
    {
        UNIT_TYPE in = input[input_idx];
        max_value = max(max_value, in);
        data[DATA_PER_WORKITEM-1] = in;
    }
    max_value = sub_group_reduce_max(max_value);

    // PART 2. Calculate DENOMINATOR
    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    ACCUMULATOR_TYPE denominator = 0.0;
    for (uint cls = 0; cls < FULL_ITERATIONS_NUM; cls++)
    {
        data[cls] = native_exp(data[cls] - max_value);
        denominator += data[cls];
    }
    if(simd_lane < LEFTOVERS)
    {
        data[DATA_PER_WORKITEM-1] = native_exp(data[DATA_PER_WORKITEM-1] - max_value);
        denominator += data[DATA_PER_WORKITEM-1];
    }

    denominator = sub_group_reduce_add(denominator);

    // PART 3. Write out results
    uint output_idx = out_depth_offset + simd_lane * OUTPUT_CLASS_PITCH;
    for (uint cls = 0; cls < FULL_ITERATIONS_NUM; cls++)
    {
        const UNIT_TYPE res = data[cls] / (UNIT_TYPE)denominator;
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
        output_idx += WORKITEMS_PER_CLASSES * OUTPUT_CLASS_PITCH;
    }
    if(simd_lane < LEFTOVERS)
    {
        const UNIT_TYPE res = data[DATA_PER_WORKITEM-1] / (UNIT_TYPE)denominator;
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
    }
}

#undef FULL_ITERATIONS_NUM
#undef DATA_PER_WORKITEM
