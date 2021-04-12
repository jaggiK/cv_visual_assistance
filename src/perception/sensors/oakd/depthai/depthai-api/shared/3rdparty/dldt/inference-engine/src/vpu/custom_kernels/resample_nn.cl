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

#define USE_OPTIMIZED_ROUND

#ifdef USE_OPTIMIZED_ROUND
    #define ROUND(x)  ((int)((x) + 0.5f))
#else
    #define ROUND(x)  (int)(round(x))
#endif

kernel void resample_nearest(__global const half* restrict src,
                             __global       half* restrict dst,
                             int iw,
                             int ih,
                             float factor,
                             int ow,
                             int oh,
                             int channels)
{
    int oy = min((int)get_global_id(0), oh-1);
    int c = get_global_id(1);
    int b = get_global_id(2);

    float fx = 1.f / factor;
    float fy = 1.f / factor;

    __global const half* start_src = src + b * iw * ih * channels + iw * ih * c;
    __global       half* start_dst = dst + b * ow * oh * channels + ow * oh * c;

    for (int ox = 0; ox < ow; ox++)
    {
        float ix_r0 = ox*fx + fx / 2.0f - 0.5f;
        float iy_r0 = oy*fy + fy / 2.0f - 0.5f;

        int ix_r1 = ROUND(ix_r0);
        int iy_r1 = ROUND(iy_r0);

        ix_r1 = max(ix_r1, 0);
        ix_r1 = min(ix_r1, iw - 1);

        iy_r1 = max(iy_r1, 0);
        iy_r1 = min(iy_r1, ih - 1);

        start_dst[oy * ow + ox] = start_src[iy_r1 * iw + ix_r1];
    }
}
