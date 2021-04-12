/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef JIT_PRIMITIVE_CONF_HPP
#define JIT_PRIMITIVE_CONF_HPP

#include <stdint.h>

#include "primitive_attr.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

/* convolution */
enum conv_version_t {ver_unused, ver_fma, ver_avx512_core, ver_4fma, ver_4vnni,
                     ver_vnni};
enum conv_loop_order_t {loop_cgn, loop_gnc, loop_ngc, loop_gncw, loop_cwgn,
                            loop_ngcw, loop_nhwcg, loop_nwcg};
enum conv_1x1_loop_order_t {loop_rbl, loop_rlb, loop_lbr, loop_lrb, loop_blr,
                            loop_brl};
enum conv_kernel_kind_t {embd_bcast, expl_bcast};
enum conv_harness_t {harness_2d_reduction, harness_3d_reduction,
                     harness_mb_reduction};

enum {
    FLAG_MB_FIRST = 1 << 0, FLAG_MB_LAST = 1 << 1,
    FLAG_OC_FIRST = 1 << 2, FLAG_OC_LAST = 1 << 3,
    FLAG_IC_FIRST = 1 << 4, FLAG_IC_LAST = 1 << 5,
    FLAG_SP_FIRST = 1 << 6, FLAG_SP_LAST = 1 << 7,
    FLAG_REDUCE_FIRST = 1<<8, FLAG_REDUCE_LAST = 1<<9,
    FLAG_ZERO_FILTER = 1 << 0, /* Controls whether the inner kernel skips
                                   loading weights-data from memory; this
                                   needs to happen on the first Group/16
                                   iteration. */
    FLAG_ZERO_BIAS = 1 << 1, /* Controls whether the inner kernel skip
                               loading bias data from memory */
    FLAG_COMPUTE_BIAS = 1 << 2, /* Controls bias computation during execution
                                    pass */
};

struct jit_conv_conf_t {
    prop_kind_t prop_kind;
    conv_version_t ver;
    conv_loop_order_t loop_order;
    conv_harness_t harness;

    int simd_w;
    int ndims;
    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int oc_padded;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    memory_format_t src_fmt;
    memory_format_t dst_fmt;
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_dw_conv;
    bool with_binarization;

    post_ops_t::entry_t::eltwise_t eltwise;

    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;

    int idp, ihp, iwp, ohp, owp;

    const float* conv_weights;
    const float* conv_biases;
    int dw_conv_oh, dw_conv_ow;
    data_type_t dw_conv_dst_dt;

    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ow, ow_block;
    int nb_oc_blocking; /* used in jit kernels for nb_oc work bloking taking
                           into account vector registers distribution */
    int nb_oc_blocking_thr_chunk; /* used for distibution of nb_oc work
                                      within threads */
    int nb_ic_blocking, nb_ic_blocking_max; // blocking of nb_ic work
    int nb_ic_L2;
    int h_blocking;
    int nb_oc_L2;
    int ur_h, ur_w;
    int ur_w_tail;
    bool is_1stconv;
    int nonblk_group_off;
    /* fma avx512_core */
    conv_kernel_kind_t kernel_kind;
    /* 4fma */
    int tr_iw;
    int tr_src_num_guard_elems;
    /* 1st conv: 4fma */
    int tr_ld;
    int kh_step;
    /* 4vnni */
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;
    int tr_ow;
    /* avx512_u8s8u8 */
    int ic_nb1, ic_nb2;
    int oc_nb1;
    int ur_ow_max, ur_ow, ur_ow_tail;
    int ur_ow_nsteps;
    data_type_t src_dt;
    data_type_t bia_dt;
    /* bf16 data-type for output */
    data_type_t dst_dt;
    data_type_t dsrc_dt;
    data_type_t dwei_dt;
    /* avx512: max possible value is nregs(32) - aux_regs(4) */
    int src_offsets[28];
    int src_count;
    bool expl_bcast;
    bool large_spatial;
    int is_oc_scale;
    int max_regs_ur; // maximum accumulation registers
    // dw conv
    int nb_ch, ch_block, nb_ch_blocking;
    bool is_depthwise, is_fast_depthwise, is_resrc_depthwise;
    int aligned_threads;
    // large spatial
    int oh_blk_size;
    // s8s8 convolution
    bool signed_input;
    int with_scales;
    bool with_input_zp;
    bool with_weights_zp;
    bool is_per_channel_input_zp;
    bool is_per_channel_weights_zp;
    float wei_adj_scale;

    bool is_cpx;

    // planar conv
    int nb_ow_blocking;

    int oh_block;
    int nb_oh_blocking;
    int oh_block_step;
};

struct jit_conv_conf_2x3_wino_t {
    conv_version_t ver;

    int m;
    int r;
    int alpha;
    int tile_h, tile_w;

    int mb;
    int ngroups, ic, oc, oc_without_padding;
    int ih, iw, oh, ow;
    int l_pad, t_pad;
    int r_pad, b_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;

    int nb_ic, ic_block;
    int nb_oc, oc_block;

    int w_block_size, h_block_size;

    data_type_t bia_dt;
    data_type_t dst_dt;

    int is_oc_scale;
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;

    memory_format_t src_fmt;
    bool with_bias;
    bool small_mb;

    int xb, yb;
    int inp_stride;
    int out_stride;
    int wei_stride;
    int bia_stride;

    int M, N, K;
    int m_block, n_block, k_block;
    int n2_block, n_chunks;
    int k2_block, k_chunks;

    int mb_block, nb_mb;

    size_t size_wino_src, size_wino_wei, size_wino_dst;

    int nthr;
};

/*
   Winograd sched policy:

   Computation Unit:
   W: weights transform
   S: src transform
   D: dst transform
   G: gemm

   Thread grouping by:
   i: nb_ic
   o: nb_oc
   t: tile_block
   e: element in tile

   Note: 'i' and 'o' are omited if
   i. not comblined with t or
   ii. with discrete transforms

   Current policies supported:
*/
enum winograd_sched_t {
    WSCHED_INVALID = 0,

    /* Forward & backward-data */
    /* W_S_G_D implements discrete transforms */
    WSCHED_DATA_W_S_G_D,
    /* W_SGD implements tiled transforms s.t. GEMM could reuse data in L2*/
    WSCHED_DATA_W_SGD,

    /* Backward-weights */
    WSCHED_WEI_S_D_G_W,
    WSCHED_WEI_SDGtWo,
    WSCHED_WEI_S_D_Giot_W,
    WSCHED_WEI_SDGt_W,
};

struct jit_conv_winograd_conf_t : public jit_conv_conf_t {
    int itiles;
    int jtiles;
    int ntiles;
    int ic_simd_block=16;
    int tile_4fma_padding;
    int tile_4fma;
    int oc_simd_block=16;
    int oc_reg_block;
    int ic_reg_block;
    int tile_block;
    int tile_block_ur;
    int nb_tile_block_ur;

    bool double_buffering;
    bool with_relu_postsum;
    int zmm_start;
    int nb_reg;

    int dimK;
    int dimK_4fma;
    int dimK_reg_block;
    int dimK_block;
    int dimK_nb_block;

    int dimM;
    int dimM_reg_block;
    int dimM_simd_block;
    int dimM_block;
    int dimM_nb_block;

    int dimN;
    int dimN_reg_block;
    int dimN_bcast_ur;
    int dimN_block;
    int dimN_nb_block;

    winograd_sched_t sched_policy;
};

struct jit_bin_conv_conf_t {
    prop_kind_t prop_kind;
    conv_version_t ver;
    conv_loop_order_t loop_order;

    int ndims;
    int mb;
    int ngroups, ic, oc, oc_padded, ic_padded;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    memory_format_t src_fmt;
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_dw_conv;
    bool with_binarization;

    float pad_value;
    bool exclude_pad;

    int dw_conv_oh;
    int dw_conv_ow;
    data_type_t dw_conv_dst_dt;

    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic
    int ur_h, ur_w;
    int ur_w_tail;
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;
    data_type_t src_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
};

struct jit_def_conv_conf_t {
    prop_kind_t prop_kind;

    int ndims;
    int mb;
    int dg;
    int ngroups, ic, oc, oc_padded, ic_padded;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    memory_format_t src_fmt;
    bool with_bias;
    bool with_sum;
    int nthr;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ic_blocking, nb_oc_blocking;
    int ur_h, ur_w;
    int ur_w_tail;
    int typesize_in;
    int typesize_off;
    int typesize_bia;
    int typesize_out;
    data_type_t src_dt;
    data_type_t off_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
};

struct jit_conv_call_s {
    const void *src; /* hack, non-const for backward_data */
    const void *dst; /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *src_prf;
    const void *dst_prf;
    const void *filt_prf;
    const void *bias_prf;
    const void *scales;
    const void *acc_s32;
    const void *compensation;
    size_t kd_offset;
    size_t kd_offset_prf;
    size_t kh_offset;
    size_t kh_offset_prf;
    size_t os_index_begin;
    size_t os_index_begin_prf;
    size_t os_index_end;
    size_t os_index_end_prf;
    size_t kd_padding;
    size_t kd_padding_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t owb;
    size_t owb_prf;
    size_t kw_padding;
    size_t channel;
    size_t channel_prf;
    size_t oc_blocks;
    size_t oc_work;
    size_t ur_w;
    size_t ur_str_w;
    size_t ch_blocks;
    size_t ch_work;
    size_t t_overflow;
    size_t b_overflow;
    size_t l_overflow;
    size_t r_overflow;
    size_t front_overflow;
    size_t back_overflow;
    size_t oh_blocks;
    int flags;

    const void *src_row0; /* hack, non-const for backward_data */
    const void *src_row1; /* hack, non-const for backward_data */
    const void *src_row2; /* hack, non-const for backward_data */

    size_t oc_off;
    size_t oc_off_prf;
    const void *input_zp;
    const void *weights_zp;
    const void *weights_zp_compensation;
};

struct jit_deconv_call_s {
    const void *src; /* hack, non-const for backward_data */
    const void *dst; /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *scales;
    const void *compensation;
    size_t t_overflow;
    size_t b_overflow;
    size_t kh_padding;
    size_t oc_blocks;
};

struct jit_dw_conv_call_s {
    const void *input;
    const void *output;
    const void *filter;
    const void *bias;
    size_t kh_count;
    size_t oh_count;
    size_t oh_index;
    size_t filter_pad_off;
    unsigned char
            exec_flags; /* Flags passed by driver execution to inner kernel */
};

struct jit_wino_transform_call_s {
    size_t tile_block;
    size_t tile_block_ur;
    size_t nb_tile_block_ur;
    size_t tile_count;
    size_t tj;
    size_t ti;
    void *src;
    void *dst;
    void *Mw;
    void *M;
    void *T;
    void *G;
    void *bias;
};

struct jit_1x1_conv_conf_t {
    prop_kind_t prop_kind;
    conv_version_t ver;

    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int iw, ih, ow, oh;
    int l_pad, t_pad;
    int r_pad, b_pad;
    int kh, kw;
    int stride_h, stride_w;
    memory_format_t src_fmt;
    memory_format_t dst_fmt;
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_dw_conv;

    post_ops_t::entry_t::eltwise_t eltwise;

    int is, os;
    int ic_block, oc_block;

    int ur, ur_tail;

    int reduce_dim, reduce_block, nb_reduce,
        nb_reduce_blocking, nb_reduce_blocking_max;
    int load_dim, load_block, nb_load,
        nb_load_blocking, nb_load_blocking_max, nb_load_chunk;
    int bcast_dim, bcast_block, nb_bcast,
        nb_bcast_blocking, nb_bcast_blocking_max;

    int reduce_loop_unroll, reduce_loop_bcast_step, reduce_loop_load_step;
    int load_loop_load_step, load_loop_iter_step;
    int bcast_loop_output_step, bcast_loop_output_substep;
    int bcast_loop_bcast_step, bcast_loop_bcast_substep;
    int fma_step;
    int load_grp_count;
    conv_1x1_loop_order_t loop_order;
    bool use_vmovntps;
    /* avx512 core */
    bool expl_bcast;
    /* 4vnni */
    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;
    /* 4fma */
    bool transpose_src;
    int tr_is;
    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
    int is_oc_scale;
    data_type_t src_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    bool signed_input;
    float wei_adj_scale;
    int dw_conv_oh, dw_conv_ow;
    data_type_t dw_conv_dst_dt;

    bool with_input_zp;
    bool with_weights_zp;
    bool is_per_channel_input_zp;
    bool is_per_channel_weights_zp;
    
    bool is_cpx;

    /* u8s8s32x */
    int ic_dim, nb_ic, nb_ic_blocking, nb_ic_blocking_max;
    int oc_dim, nb_oc, nb_oc_blocking, nb_oc_blocking_max;
    int is_dim, os_block, nb_oh_blocking, nb_oh_blocking_max;
    int ow_tail;

    int ic_loop_unroll, ic_loop_src_step, ic_loop_wei_step;
    int os_loop_dst_step, os_loop_src_step, os_loop_acc_step;
    int os_loop_src_tail_step, os_loop_dst_tail_step, os_loop_acc_tail_step;
};

struct jit_gemm_conv_conf_t {
    prop_kind_t prop_kind;

    int mb;
    int ngroups, ic, oc;
    int iw, ih, id, ow, oh, od;
    int l_pad, t_pad, f_pad;
    int kh, kw, kd;
    int stride_h, stride_w, stride_d;
    int dilate_h, dilate_w, dilate_d;
    memory_format_t src_fmt;
    bool with_bias;

    int is, os, ks;
    int ic_block, oc_block;

    int nthr;
    ptrdiff_t im2col_sz;
    bool need_wei_reduction;
    bool signed_input;
    float wei_adj_scale;
    int oh_block;
    int ow_block;
    bool outer_threading;

    bool with_input_zp;
    bool with_weights_zp;
};

struct jit_1x1_conv_call_s {
    const void *bcast_data;
    const void *load_data;
    const void *output_data;
    const void *bias_data; // used in forward and backward_weights only
    const void *acc_s32;
    const void *scales;
    const void *compensation;

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;

    size_t output_stride; // used in backward_weights only

    size_t first_last_flag;

    // dw conv fusing
    const void *weights_dw;
    const void *bias_dw;

    size_t oc_off;

    /* u8s8s32x */
    size_t oc_dim;
    size_t os_dim;
    size_t ic_dim;
    size_t ic_pos_flag;
	const void *is_data;
	const void *oc_data;
};

struct jit_def_conv_call_s {
    const void *src;
    const void *off;
    const void *filt;
    const void *bias;
    const void *dst;
    const void *buf;
    size_t oh_pos;
};

/* pooling */
struct jit_pool_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw, od, oh, ow;
    int stride_d, stride_h, stride_w;
    int kd, kh, kw;
    int f_pad, t_pad, l_pad, b_pad, r_pad, back_pad;
    alg_kind_t alg;
    bool is_training;
    bool pad_w_is_null;
    bool is_backward;
    bool simple_alg;
    data_type_t ind_dt;

    int c_block, c_tail, nb_c;
    int ur_c, ur_c_tail;
    int ur_w;
    int ur_w_tail;
    size_t tail[4];
    data_type_t src_dt;
    data_type_t dst_dt;

    bool is_bf16;
    int dt_size;

    bool is_cpx;
};

struct jit_pool_call_s {
    const void *src;
    const void *dst;
    const void *indices;
    const void *src_prf;
    const void *dst_prf;
    const void *indices_prf;
    size_t oh;
    size_t kd_padding;
    size_t kh_padding;
    size_t kh_padding_shift;
    size_t kd_padding_shift;
    size_t kw_padding;
    const void *init_value;
    float ker_area_h;
};

/* roi pooling */
struct jit_roi_pool_conf_t {
    int mb, c;
    int ih, iw, oh, ow;

    int c_block, nb_c, nb_c_blocking;

    double spatial_scale;
    int pooled_h;
    int pooled_w;

    alg_kind_t alg;
};

struct jit_roi_pool_call_s {
    const float *src;
    float *dst;

    size_t kh;
    size_t kw;
    size_t bin_area;

    size_t c_blocks;

    float xf;
    float yf;

    size_t xoff;
    size_t yoff;
};

/* softmax */
struct jit_softmax_conf_t {
    size_t outer_size;
    size_t channels;
    size_t inner_size;
    size_t ur_channel;
    size_t ur_inner;
    size_t outer_block;
};

struct jit_softmax_call_s {
    const float *src;
    float *dst;

    size_t channels;
    size_t work;
};

}
}
}

#endif
