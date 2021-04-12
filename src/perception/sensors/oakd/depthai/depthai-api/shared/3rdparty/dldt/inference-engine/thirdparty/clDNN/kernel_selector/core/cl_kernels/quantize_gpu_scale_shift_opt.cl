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

#include "include/common.cl"
#include "include/data_types.cl"
#include "include/fetch.cl"

KERNEL(quantize_gpu_scale_shift_opt)(const __global INPUT0_TYPE* input,
                                     const __global INPUT1_TYPE* input_low,
                                     const __global INPUT2_TYPE* input_high,
                                     const __global INPUT3_TYPE* output_low,
                                     const __global INPUT4_TYPE* output_high,
                                     const __global INPUT5_TYPE* input_scale,
                                     const __global INPUT6_TYPE* input_shift,
                                     const __global INPUT7_TYPE* output_scale,
                                     const __global INPUT8_TYPE* output_shift,
                                           __global OUTPUT_TYPE* output)
{
    const int b = get_global_id(GWS_BATCH);
    const int of = get_global_id(GWS_FEATURE);
#if OUTPUT_DIMS <= 4
    const int yx = get_global_id(GWS_YX);
    const int x = yx % OUTPUT_SIZE_X;
    const int y = yx / OUTPUT_SIZE_X;
    const int z = 0;
#elif OUTPUT_DIMS == 5
    const int zyx = get_global_id(GWS_YX);
    const int x = zyx % OUTPUT_SIZE_X;
    const int y = (zyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const int z = (zyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
#endif

#if INPUT0_DIMS == 5
    const int input_offset = INPUT0_GET_INDEX(b, of, z, y, x);
#elif INPUT0_DIMS <= 4
    const int input_offset = INPUT0_GET_INDEX(b, of, y, x);
#endif
#if OUTPUT_DIMS == 5
    const int output_offset = OUTPUT_GET_INDEX(b, of, z, y, x);
#elif OUTPUT_DIMS <= 4
    const int output_offset = OUTPUT_GET_INDEX(b, of, y, x);
#endif

#if INPUT1_DIMS == 4
    const int in_range_offset = INPUT1_GET_INDEX_SAFE(b, of, y, x);
#elif INPUT1_DIMS == 5
    const int in_range_offset = INPUT1_GET_INDEX_SAFE(b, of, z, y, x);
#endif

#if INPUT7_DIMS == 4
    const int scales_offset = INPUT7_GET_INDEX_SAFE(b, of, y, x);
#elif INPUT7_DIMS == 5
    const int scales_offset = INPUT7_GET_INDEX_SAFE(b, of, z, y, x);
#endif

    INPUT1_TYPE input_scale_val  = input_scale[scales_offset];
    INPUT1_TYPE input_shift_val  = input_shift[scales_offset];
    INPUT1_TYPE output_scale_val = output_scale[scales_offset];
    INPUT1_TYPE output_shift_val = output_shift[scales_offset];
    INPUT1_TYPE input_low_val    = input_low[in_range_offset];
    INPUT1_TYPE input_high_val   = input_high[in_range_offset];
    INPUT1_TYPE val = min(max(TO_INPUT1_TYPE(input[input_offset]),input_low_val), input_high_val);
#if OUTPUT_IS_FP
    output[output_offset] = TO_OUTPUT_TYPE_SAT(round(val * input_scale_val + input_shift_val) * output_scale_val + output_shift_val);
#else
    // TODO: the outer round should be deleted once output range is correct
    output[output_offset] = TO_OUTPUT_TYPE_SAT(round(round(val * input_scale_val + input_shift_val) * output_scale_val + output_shift_val));
#endif
}
