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

#include "include/common.cl"
#include "include/data_types.cl"


// ---------------------------------------------------------------------------------------------------------------------
// Just-in-time macro definitions:
// ---------------------------------------------------------------------------------------------------------------------

// Required JIT constants:
//  - INPUT                - [tensor] Input dimensions (batch, spatial and feature).
//  - OUTPUT               - [tensor] Output dimensions (batch, spatial and feature).
//  - STRIDE               - [tensor] Stride (only spatial). Factors that describe step size in X or Y dimension of
//                           input position of application of convolution filter when next ouput value
//                           (step 1 in in X or Y dimension of output) is computed.
//  - INPUT0_OFFSET        - [tensor] Offset for the first element
//                           initial offset input position of application of convolution filter and output position.
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).
//
//  - SUB_GROUP_SIZE       - [int] Size of used subgroup (SIMD).
//  - LEFTOVERS            - [int] Optional parameter, required only when number of ofm is not dividable by SUB_GROUP_SIZE
//                           see comment for FEATURES_THREADS_PER_BATCH for more informations

/*
gpu::make_jit_constant("OUTPUT_LIMIT",              output_size),
gpu::make_jit_constant("FILTER",                    filter_mem.argument().size),
gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
gpu::make_jit_constant("OUTPUT_BLOCK_WIDTH",        _kernel_data.block_width));
gpu::make_jit_constant("OUTPUT_BLOCK_HEIGHT",       _kernel_data.block_height));
gpu::make_jit_constant("IN_BLOCK_ARRAY_SIZE",       _kernel_data.input_block_array_size));
gpu::make_jit_constant("IN_BLOCK_WIDTH",            _kernel_data.input_block_width));
gpu::make_jit_constant("PREFETCH",                  _kernel_data.prefetch));
if (_kernel_data.leftovers)
    gpu::make_jit_constant("LEFTOVERS",             _kernel_data.leftovers));
*/

// FEATURES_THREADS_PER_BATCH defines how many threads in z-dimension are processing single batch.
// ideally, z-dimension of value n should indicate processing of n-th output feature. however, since
// threads are stack in groups of SUB_GROUP_SIZE, when number of ofm is not dividable by SUB_GROUP_SIZE
// there are dummy threads added in z-dimension in count of LEFTOVERS. We need to take them into consideration
// while calculating batch's id (see lines 86-87). Values calculated by dummy threads are discarded at line 210.
#ifdef LEFTOVERS
#define FEATURES_THREADS_PER_BATCH (FILTER_OFM_NUM + LEFTOVERS)
#else
#define FEATURES_THREADS_PER_BATCH (FILTER_OFM_NUM)
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(convolution_gpu_bfyx_os_iyx_osv16)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weights,
#if BIAS_TERM
    const __global UNIT_TYPE* bias,
#endif
    uint split_idx,
    const __global UNIT_TYPE* eltw_input) // TODO: removing this parameter cause a performance degradation... :)
{
    const uint oc  = (uint)get_global_id(0) * OUTPUT_BLOCK_WIDTH;  // oc = Output Column
    const uint or  = (uint)get_global_id(1) * OUTPUT_BLOCK_HEIGHT; // or = Output Row
    const uint fm  = get_global_id(2);                    // fm = Feature Map = od = Output Depth
    const uint lid = get_sub_group_local_id();

    uint batch_idx = fm / FEATURES_THREADS_PER_BATCH;
    uint feature_idx = fm % FEATURES_THREADS_PER_BATCH;
    uint fmg = feature_idx / SUB_GROUP_SIZE;

    UNIT_TYPE in[IN_BLOCK_ARRAY_SIZE];
    UNIT_TYPE out[OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT];
    UNIT_TYPE w[PREFETCH];
    uint in_addr;
    uint weight_addr = fmg * FILTER_IFM_NUM * FILTER_SIZE_X * FILTER_SIZE_Y * SUB_GROUP_SIZE + lid;

    for(int i = 0; i < (OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT); i++) {
        out[i] = UNIT_VAL_ZERO;
    }

    uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    in_addr = batch_idx * INPUT0_BATCH_PITCH;
    in_addr += in_split_offset + INPUT0_OFFSET_WITH_PADDING + or * STRIDE_SIZE_Y * INPUT0_Y_PITCH + oc * STRIDE_SIZE_X + lid;

    for(int kd = 0; kd < FILTER_IFM_NUM; kd++)  // _ID = 3, RGB
    {
        uint tmp_in_addr = in_addr;

#if IN_BLOCK_WIDTH % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;

            in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + in_block_pos % IN_BLOCK_WIDTH];

            // If we have row break, move to the next row.
            if (in_block_next_x_pos == IN_BLOCK_WIDTH)
                tmp_in_addr += INPUT0_Y_PITCH;
        }
#elif (2 * IN_BLOCK_WIDTH) % SUB_GROUP_SIZE == 0
        __attribute__((opencl_unroll_hint(IN_BLOCK_ARRAY_SIZE)))
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE * SUB_GROUP_SIZE; in_block_pos += SUB_GROUP_SIZE) {
            // Horizontal position in input block after read.
            const uint in_block_next_x_pos = in_block_pos % IN_BLOCK_WIDTH + SUB_GROUP_SIZE;

            if (in_block_next_x_pos <= IN_BLOCK_WIDTH) { //
                in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + in_block_pos % IN_BLOCK_WIDTH];

                // If we have row break, move to the next row.
                if (in_block_next_x_pos == IN_BLOCK_WIDTH)
                    tmp_in_addr += INPUT0_Y_PITCH;
            }
            else {
                // TODO: Generalize this step to relax IN_BLOCK_WIDTH restrictions.
                // Position in sub-group on which new row need to be read.
                const uint sg_br_pos = IN_BLOCK_WIDTH - in_block_pos % IN_BLOCK_WIDTH;

                if (lid < sg_br_pos)
                    in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr + in_block_pos % IN_BLOCK_WIDTH];
                // We have row break inside sub-group. Need to move to next line.
                tmp_in_addr += INPUT0_Y_PITCH;
                if (lid >= sg_br_pos)
                    in[in_block_pos / SUB_GROUP_SIZE] = input[tmp_in_addr - sg_br_pos];

                // If we have another row break, move to the next row.
                if (in_block_next_x_pos == 2 * IN_BLOCK_WIDTH)
                    tmp_in_addr += INPUT0_Y_PITCH;
            }
        }
#else
    #error IN_BLOCK_WIDTH must be multiple of SUB_GROUP_SIZE or half of SUB_GROUP_SIZE. Other scenarios are not currently implemented.
#endif

        //move to next filter
        in_addr += INPUT0_FEATURE_PITCH;

        for(int pf=0; pf<PREFETCH; pf++) {
            w[pf] = weights[weight_addr]; weight_addr += SUB_GROUP_SIZE;
        }

        uint wi = 0;
        uint kr = 0; // kr = Kernel Row
        LOOP(FILTER_SIZE_Y, kr,  // LOOP is a macro that unrolls the loop.
        {
            uint kc = 0; // kc = Kernel Column
            LOOP(FILTER_SIZE_X, kc,
            {
                //w = weights[weight_addr];
                for(uint br=0; br<OUTPUT_BLOCK_HEIGHT; br++) {
                    for(uint bc=0; bc<OUTPUT_BLOCK_WIDTH; bc++) {

#if IN_BLOCK_WIDTH != SUB_GROUP_SIZE
                        //if we fix the programming model, then we could use a nice simple 2d array: val = in[br * STRIDE_SIZE_Y + kr][bc * STRIDE_SIZE_X + kc];
                        UNIT_TYPE val = intel_sub_group_shuffle( in[(((br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y) * IN_BLOCK_WIDTH) + (bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X)) / SUB_GROUP_SIZE],
                                                                    (((br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y) * IN_BLOCK_WIDTH) + (bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X)) % SUB_GROUP_SIZE);
#else
                        UNIT_TYPE val = intel_sub_group_shuffle( in[br * STRIDE_SIZE_Y + kr * DILATION_SIZE_Y], bc * STRIDE_SIZE_X + kc * DILATION_SIZE_X);
#endif

                        out[br * OUTPUT_BLOCK_WIDTH + bc] = mad(w[wi % PREFETCH], val, out[br * OUTPUT_BLOCK_WIDTH + bc]);
                    }
                }
                w[wi % PREFETCH] = weights[weight_addr];
                weight_addr += SUB_GROUP_SIZE; // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                wi++;
            });
        });
        // addr went beyond due to prefetch so move it back to correct location.
        weight_addr -= PREFETCH * SUB_GROUP_SIZE;
    }

    uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    uint out_addr = OUTPUT_OFFSET;
    out_addr += batch_idx * OUTPUT_BATCH_PITCH;
    out_addr += out_split_offset + feature_idx * OUTPUT_FEATURE_PITCH; // out_addr indices into start of 16 feature maps.
    out_addr += or * OUTPUT_Y_PITCH + oc;  // offset for the 4x3 block that this workitem is working on;

#if BIAS_TERM
    for(uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
#if BIAS_PER_OUTPUT
            const unsigned bias_index = feature_idx*OUTPUT_SIZE_X*OUTPUT_SIZE_Y + or*OUTPUT_SIZE_X + oc;
#else
            const unsigned bias_index = feature_idx;
#endif
            out[r * OUTPUT_BLOCK_WIDTH + c] += bias[bias_index];
        }
    }
#endif


    for(uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
            out[r * OUTPUT_BLOCK_WIDTH + c] = ACTIVATION_CONV(out[r * OUTPUT_BLOCK_WIDTH + c], ACTIVATION_PARAMS_CONV);
        }
    }

#if IN_OUT_OPT != 1
    // eltwise part
    uint eltw_addr = INPUT1_OFFSET;
    eltw_addr += batch_idx * INPUT1_BATCH_PITCH;
    eltw_addr += out_split_offset + feature_idx * INPUT1_FEATURE_PITCH; // eltw_addr indices into start of 16 feature maps.
    eltw_addr += (or * ELTW_STRIDE_Y) * INPUT1_Y_PITCH + (oc * ELTW_STRIDE_X);  // offset for the 4x3 block that this workitem is working on;

    for(uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        for(uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
            out[r * OUTPUT_BLOCK_WIDTH + c] += eltw_input[eltw_addr + r * INPUT1_Y_PITCH * ELTW_STRIDE_Y + c * ELTW_STRIDE_X];
            out[r * OUTPUT_BLOCK_WIDTH + c] = ACTIVATION_ELTW(out[r * OUTPUT_BLOCK_WIDTH + c], ACTIVATION_PARAMS_ELTW);
        }
    }
    // end of eltwise part
#endif

#ifdef LEFTOVERS
    if (feature_idx < OUTPUT_FEATURE_NUM)
#endif
    for(uint r = 0; r < OUTPUT_BLOCK_HEIGHT; r++) {
        if(!(or + r >= OUTPUT_SIZE_Y))
        {
            for(uint c = 0; c < OUTPUT_BLOCK_WIDTH; c++) {
                // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
                if(!(oc + c >= OUTPUT_SIZE_X))
                {
#if IN_OUT_OPT == 1
                    out[r * OUTPUT_BLOCK_WIDTH + c] += output[out_addr + r * OUTPUT_Y_PITCH + c];
                    out[r * OUTPUT_BLOCK_WIDTH + c] = ACTIVATION_ELTW(out[r * OUTPUT_BLOCK_WIDTH + c], ACTIVATION_PARAMS_ELTW);
#endif
                    output[out_addr + r * OUTPUT_Y_PITCH + c] = out[r * OUTPUT_BLOCK_WIDTH + c];
                }
            }
        }
    }
}

#undef FEATURES_THREADS_PER_BATCH
