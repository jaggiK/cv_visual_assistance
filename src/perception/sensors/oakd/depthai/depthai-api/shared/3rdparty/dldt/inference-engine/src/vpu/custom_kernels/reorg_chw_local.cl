// Copyright (C) 2018-2020 Intel Corporation
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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// kernel with local memory buffer
__kernel void reorg(__global const half* restrict src,
                    __global       half* restrict out,
                    __local        half* restrict tmp,
                    int H,
                    int W,
                    int stride)
{
    int h = min((int)get_global_id(0), H-1);

    int c = get_global_id(1);
    int C = get_global_size(1);
    int C2 = C/(stride*stride);

    int offset = c / C2;

    int c2 = c - C2 * offset;

    int H2 = H*stride;
    int W2 = W*stride;

    for (int w = 0; w < W; ++w)
    {
        int h2 = h*stride + offset / stride;
        int w2 = w*stride + offset - stride * (offset / stride);

        tmp[get_local_id(1)*get_local_size(0)*W + get_local_id(0)*W + w] = src[W2*H2*c2 + W2*h2 + w2];
    }

    for (int w = 0; w < W; ++w)
    {
        out[W*H*c + W*h + w] = tmp[get_local_id(1)*get_local_size(0)*W + get_local_id(0)*W + w];
    }
}
