/*
// Copyright (c) 2016 Intel Corporation
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

// ! Huge warning !
// Reshape functions below assume both input and output formats are bfyx, bfzyx, bfwzyx, etc.

inline uint8 FUNC(reshape_2_to_4)(uint o, uint i, uint y, uint x, uint dst_size_y, uint dst_size_x)
{
    uint _i  = i / (dst_size_y*dst_size_x);
    uint _yx = i % (dst_size_y*dst_size_x);
    uint _y = _yx / dst_size_x;
    uint _x = _yx % dst_size_x;
    return (uint8)(o,_i, 0, 0, _y,_x, 0, 0);
}

inline uint8 FUNC(reshape_4_to_2)(uint o, uint i, uint y, uint x, uint src_size_y, uint src_size_x)
{
    uint _i = i*src_size_y*src_size_x + y*src_size_x + x;
    return (uint8)(o, _i, 0, 0, 0, 0, 0, 0);
}

inline uint8 FUNC(reshape_4_to_5)(uint o, uint i, uint y, uint x,
    uint src_size_f, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_f = src_pitch_y * src_size_y;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_z = flat_idx % dst_size_z;
    flat_idx /= dst_size_z;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(dst_b, dst_f, 0, dst_z, dst_y, dst_x, 0, 0);
}

inline uint8 FUNC(reshape_5_to_4)(uint o, uint i, uint z, uint y, uint x,
    uint src_size_f, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_z = src_pitch_y * src_size_y;
    const uint src_pitch_f = src_pitch_z * src_size_z;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(dst_b, dst_f, 0, 0, dst_y, dst_x, 0, 0);
}

inline uint8 FUNC(reshape_4_to_6)(uint o, uint i, uint y, uint x,
    uint src_size_f, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_f = src_pitch_y * src_size_y;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_z = flat_idx % dst_size_z;
    flat_idx /= dst_size_z;
    uint dst_w = flat_idx % dst_size_w;
    flat_idx /= dst_size_w;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(dst_b, dst_f, dst_w, dst_z, dst_y, dst_x, 0, 0);
}

inline uint8 FUNC(reshape_6_to_4)(uint o, uint i, uint w, uint z, uint y, uint x,
    uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
    const uint src_pitch_x = 1;
    const uint src_pitch_y = src_pitch_x * src_size_x;
    const uint src_pitch_z = src_pitch_y * src_size_y;
    const uint src_pitch_w = src_pitch_z * src_size_z;
    const uint src_pitch_f = src_pitch_w * src_size_w;
    const uint src_pitch_b = src_pitch_f * src_size_f;

    uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + w * src_pitch_w + i * src_pitch_f + o * src_pitch_b;

    uint dst_x = flat_idx % dst_size_x;
    flat_idx /= dst_size_x;
    uint dst_y = flat_idx % dst_size_y;
    flat_idx /= dst_size_y;
    uint dst_f = flat_idx % dst_size_f;
    flat_idx /= dst_size_f;
    uint dst_b = flat_idx;
    return (uint8)(dst_b, dst_f, 0, 0, dst_y, dst_x, 0, 0);
}

inline uint8 FUNC(reshape_dims)(
    uint o, uint i, uint w, uint z, uint y, uint x,
    uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
    uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
    uint src_dims, uint dst_dims)
{
    if (src_dims == 4 && dst_dims == 2)
    {
        return FUNC_CALL(reshape_4_to_2)(o,i,y,x,src_size_y,src_size_x);
    }
    else if (src_dims == 2 && dst_dims == 4)
    {
        return FUNC_CALL(reshape_2_to_4)(o,i,y,x,dst_size_y,dst_size_x);
    }
    else if (src_dims == 4 && dst_dims == 6)
    {
        return FUNC_CALL(reshape_4_to_6)(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_w, dst_size_z, dst_size_y, dst_size_x);
    }
    else if (src_dims == 6 && dst_dims == 4)
    {
        return FUNC_CALL(reshape_6_to_4)(o, i, w, z, y, x, src_size_f, src_size_w, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
    }
    else if (src_dims == 4 && dst_dims == 5)
    {
        return FUNC_CALL(reshape_4_to_5)(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_z, dst_size_y, dst_size_x);
    }
    else if (src_dims == 5 && dst_dims == 4)
    {
        return FUNC_CALL(reshape_5_to_4)(o, i, z, y, x, src_size_f, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
    }

    return (uint8)(o, i, w, z, y, x, 0, 0);
}

// ======================================================================
// Reshape dimensions from indices in `src_prefix` DataTensor to indices
// in `dst_prefix` DataTensor.
// Reshaping assumes dimension order: b, f, reverse spatial - w, z, y, x.
#define RESHAPE_DIMS(src_prefix, dst_prefix, o, i, w, z, y, x)          \
    FUNC_CALL(reshape_dims)(                                            \
        o, i, w, z, y, x,                                               \
        CAT(src_prefix, _FEATURE_NUM),                                  \
        CAT(src_prefix, _SIZE_W),                                       \
        CAT(src_prefix, _SIZE_Z),                                       \
        CAT(src_prefix, _SIZE_Y),                                       \
        CAT(src_prefix, _SIZE_X),                                       \
        CAT(dst_prefix, _FEATURE_NUM),                                  \
        CAT(dst_prefix, _SIZE_W),                                       \
        CAT(dst_prefix, _SIZE_Z),                                       \
        CAT(dst_prefix, _SIZE_Y),                                       \
        CAT(dst_prefix, _SIZE_X),                                       \
        CAT(src_prefix, _DIMS),                                         \
        CAT(dst_prefix, _DIMS))

// ======================================================================
// Reshape indices between WeightTensors.
// Weight tensor has _IFM in place of _FEATURE_NUM and does not support 6D,
// so macro for DataTensors cannot be used directly.
#define RESHAPE_WEIGHT_DIMS(src_prefix, dst_prefix, o, i, w, z, y, x)   \
    FUNC_CALL(reshape_dims)(                                            \
        o, i, w, z, y, x,                                               \
        CAT(src_prefix, _IFM_NUM),                                      \
        1,                                                              \
        CAT(src_prefix, _SIZE_Z),                                       \
        CAT(src_prefix, _SIZE_Y),                                       \
        CAT(src_prefix, _SIZE_X),                                       \
        CAT(dst_prefix, _IFM_NUM),                                      \
        1,                                                              \
        CAT(dst_prefix, _SIZE_Z),                                       \
        CAT(dst_prefix, _SIZE_Y),                                       \
        CAT(dst_prefix, _SIZE_X),                                       \
        CAT(src_prefix, _DIMS),                                         \
        CAT(dst_prefix, _DIMS))

inline uint8 FUNC(reshape_dims3d)(uint o, uint i, uint z, uint y, uint x, uint src_size_z, uint src_size_y, uint src_size_x, uint dst_size_z, uint dst_size_y, uint dst_size_x, uint src_dims, uint dst_dims)
{
    if (src_dims == 4 && dst_dims == 5)
    {
        return (uint8)(o,i,1,y,x,0,0,0);
    }
    else if (src_dims == 5 && dst_dims == 4)
    {
        uint _y = z*src_size_y + y;
        return (uint8)(o,i,0,_y,x,0,0,0);
    }
    return (uint8)(o,i,z,y,x,0,0,0);
}
