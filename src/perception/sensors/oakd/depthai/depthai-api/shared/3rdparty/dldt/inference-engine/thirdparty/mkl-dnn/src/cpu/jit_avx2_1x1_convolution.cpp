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

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include <cstring>
#include "jit_generator.hpp"

#include "jit_avx2_1x1_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

#define data_blk_off(f, n, c, h, w) \
    ((ndims == 3) \
    ? (f).blk_off(n, c, w) \
    : (f).blk_off(n, c, h, w))

/* convolution forward */

void jit_avx2_1x1_convolution_fwd_t::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    auto rtus_space = scratchpad().get<data_t>(key_conv_rtus_space);

    const auto &jcp = kernel_->jcp;
    const int MB = pd()->MB();

    const int work_amount = MB * jcp.ngroups * jcp.nb_bcast;
    const int ndims = dst_d.ndims();

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto ker = [&](const int ithr, const int nthr) {
        // TODO (Roma): remove this restriction
        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

	    auto p = jit_1x1_conv_call_s();
	    auto rp = rtus_driver_t<avx2>::call_params_t();

        const int nb_oc = jcp.nb_load;
        const int nb_ic = jcp.nb_reduce;
        const int nb_ic_blocking = jcp.nb_reduce_blocking;
        const int os_block = jcp.bcast_block;

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int iwork = start;
        while (iwork < end) {
            int n{0}, g{0}, osb{0};
            nd_iterator_init(iwork, n, MB, g, jcp.ngroups, osb,
                    jcp.nb_bcast);

            int bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                    jcp.nb_bcast_blocking_max);
            bcast_step = nstl::min(bcast_step, end - iwork);

            const int os = osb * os_block;
            const int oh = os / jcp.ow;
            const int ow = os % jcp.ow;

            const int ih = nstl::max(oh * stride_h - pad_t, 0);
            const int iw = nstl::max(ow * stride_w - pad_l, 0);
            rp.iw_start = iw;

            p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
            rp.os = p.bcast_dim;

            int ocb = 0;
            while (ocb < jcp.nb_load) {
                const int load_step = step(jcp.nb_load_blocking,
                        jcp.nb_load - ocb, jcp.nb_load_blocking_max);

                const int _ocb = g * nb_oc + ocb;
                p.load_dim = this_block_size(ocb * jcp.oc_block, jcp.oc,
                        load_step * jcp.oc_block);
                const size_t dst_off = data_blk_off(dst_d, n, _ocb, oh, ow);

                p.output_data = &dst[dst_off];

                p.bias_data = &bias[_ocb * jcp.oc_block];

                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    p.first_last_flag = 0
                        | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                        | (icb + nb_ic_blocking >= nb_ic
                                ? FLAG_REDUCE_LAST : 0);

                    p.reduce_dim = this_block_size(icb * jcp.ic_block, jcp.ic,
                            nb_ic_blocking * jcp.ic_block);
                    rp.icb = p.reduce_dim / jcp.reduce_block;

                    p.load_data = &weights[pd()->with_groups()
                        ? weights_d.blk_off(g, ocb, icb)
                        : weights_d.blk_off(ocb, icb)];

                    const int _icb = g * nb_ic + icb;
                    if (pd()->rtus_.reduce_src_) {
                        rp.ws = rtus_space
                            + ithr * pd()->rtus_.space_per_thread_
                            + _icb * jcp.is * jcp.ic_block;

                        if (ocb == 0) {
                            rp.src = src + data_blk_off(src_d, n, _icb, ih, iw);
                            rtus_driver_->ker_(&rp);
                        }

                        p.bcast_data = rp.ws;
                    } else
                        p.bcast_data = src + data_blk_off(src_d, n, _icb, ih, iw);

                    p.oc_off = _ocb * jcp.oc_block * sizeof(float);

                    kernel_->jit_ker(&p);
                }

                ocb += load_step;
            }

            iwork += bcast_step;
        }
    };

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad().get<data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(0, (size_t)work_amount, ker);

    if (pd()->wants_zero_pad_dst())
        output_memory_primitive(0)->zero_pad();
}

void jit_avx2_1x1_convolution_fwd_t::execute_forward_with_dw_conv() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    auto rtus_space = scratchpad().get<data_t>(key_conv_rtus_space);

    const auto &jcp = kernel_->jcp;
    const auto &jcp_dw = kernel_dw_->jcp;
    const int MB = pd()->MB();

    auto dw_bias = jcp_dw.conv_biases;

    int ocb_work = jcp.with_dw_conv ? utils::div_up(jcp.nb_load, jcp.nb_load_blocking) : 1;
    const int work_amount = MB * jcp.ngroups * ocb_work * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto ker = [&](const int ithr, const int nthr) {
        // TODO (Roma): remove this restriction
        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

        auto compute_block_1x1 = [&](float* ws_p, int n, int g, int oh, int ow, int ih, int iw, int os, int os_block, int bcast_step, int ocb, int load_step,
                                    int num_rows) {
            auto rp = rtus_driver_t<avx2>::call_params_t();
            auto p = jit_1x1_conv_call_s();

            for (int h = 0; h < num_rows; h++) {
                ih = nstl::max((oh + h) * jcp.stride_h - jcp.t_pad, 0);

                if ((oh + h) < 0 || (oh + h) >= jcp.ih) {
                    for (int chb = ocb; chb < ocb + load_step; chb++) {
                        memset(ws_p + (((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block +
                               (chb - ocb) * jcp_dw.kh * jcp.ow * jcp.oc_block, 0, jcp.ow * jcp.oc_block * sizeof(float));
                    }
                } else {
                    const int _ocb = g * jcp.nb_load + ocb;

                    rp.iw_start = iw;
                    p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);

                    rp.os = p.bcast_dim;
                    p.load_dim = this_block_size(ocb * jcp.oc_block, jcp.oc, load_step * jcp.oc_block);

                    p.output_data = &ws_p[(((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block];

                    p.bias_data = &bias[_ocb * jcp.oc_block];

                    for (int icb = 0; icb < jcp.nb_reduce; icb += jcp.nb_reduce_blocking) {
                        p.first_last_flag = 0
                                            | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                                            | (icb + jcp.nb_reduce_blocking >= jcp.nb_reduce
                                               ? FLAG_REDUCE_LAST : 0);

                        p.reduce_dim = this_block_size(icb * jcp.ic_block, jcp.ic,
                                                       jcp.nb_reduce_blocking * jcp.ic_block);
                        rp.icb = p.reduce_dim / jcp.reduce_block;

                        p.load_data = &weights[pd()->with_groups()
                                               ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

                        const int _icb = g * jcp.nb_reduce + icb;
                        if (pd()->rtus_.reduce_src_) {
                            rp.ws = rtus_space
                                    + ithr * pd()->rtus_.space_per_thread_
                                    + _icb * jcp.is * jcp.ic_block;

                            if (ocb == 0) {
                                rp.src = src + src_d.blk_off(n, _icb, ih, iw);
                                rtus_driver_->ker_(&rp);
                            }

                            p.bcast_data = rp.ws;
                        } else {
                            p.bcast_data = src + src_d.blk_off(n, _icb, ih, iw);
                        }

                        p.oc_off = _ocb * jcp.oc_block * sizeof(float);

                        kernel_->jit_ker(&p);
                    }
                }
            }
        };

        auto compute_row_dw = [&](const float* ws_p, int n, int ocb, int load_step, int dst_idx) {

            for (int chb = ocb; chb < ocb + load_step; chb++) {
                auto par_conv_dw = jit_conv_call_s();

                par_conv_dw.src_row0 = &ws_p[(((dst_idx+1) - 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row1 = &ws_p[(((dst_idx+1) - 0) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row2 = &ws_p[(((dst_idx+1) + 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];

                par_conv_dw.dst = &dst[n*jcp_dw.oc*jcp_dw.oh*jcp_dw.ow + chb*jcp_dw.ch_block*jcp_dw.oh*jcp_dw.ow +
                                       dst_idx/jcp_dw.stride_h*jcp_dw.ow*jcp_dw.ch_block];

                par_conv_dw.kh_padding = jcp_dw.kh;
                par_conv_dw.filt = &jcp_dw.conv_weights[chb * jcp_dw.kh * jcp_dw.kw * jcp_dw.ch_block];
                par_conv_dw.bias = &dw_bias[chb * jcp_dw.ch_block];
                par_conv_dw.ur_w = (size_t)(jcp_dw.ow);
                par_conv_dw.oc_work = nstl::min((chb + 1) * jcp_dw.ch_block, (int)jcp_dw.oc) - chb*jcp_dw.ch_block;
                par_conv_dw.oc_off = chb * jcp_dw.ch_block * sizeof(float);

                kernel_dw_->jit_ker(&par_conv_dw);
            }
        };

        assert(jcp.stride_w == 1 && jcp.stride_h == 1);

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto dw_conv_buffer = scratchpad().get<data_t>(key_dw_conv_buffer);
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * (jcp.oc / jcp.oc_block);
        auto pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;

        const int os_block = jcp.iw;

        int iwork = start;
        while (iwork < end) {
            int n{0}, g{0}, ocbb{0}, osb{0};
            nd_iterator_init(iwork, n, MB, g, jcp.ngroups, ocbb, ocb_work, osb,
                             jcp.nb_bcast);
            int bcast_step = 1;

            const int os = osb * os_block;
            const int oh = os / jcp.ow;
            const int ow = os % jcp.ow;

            const int ih = nstl::max(oh * jcp.stride_h - jcp.t_pad, 0);
            const int iw = nstl::max(ow * jcp.stride_w - jcp.l_pad, 0);

            int ocb = ocbb * jcp.nb_load_blocking;

            const int load_step = step(jcp.nb_load_blocking,
                                       jcp.nb_load - ocb, jcp.nb_load_blocking_max);

            if (iwork == start || oh == 0) {
                bcast_step = nstl::min(1, end - iwork);
                compute_block_1x1(pbuf, n, g, oh - 1, ow, ih, iw, os, os_block, bcast_step, ocb, load_step, bcast_step + 2);
            } else {
                bcast_step = nstl::min(1, end - iwork);
                compute_block_1x1(pbuf, n, g, oh + 1, ow, ih, iw, os, os_block, bcast_step, ocb, load_step, bcast_step);
            }

            if ((oh % jcp_dw.stride_h == 0)) {
                compute_row_dw(pbuf, n, ocb, load_step, oh);
            }

            iwork += bcast_step;
        }
    };

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad().get<data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;

        auto dw_padded_bias = scratchpad().get<data_t>(key_dw_conv_padded_bias);
        utils::array_copy(dw_padded_bias, dw_bias, jcp.oc_without_padding);
        utils::array_set(dw_padded_bias + jcp.oc_without_padding, 0.f,
                         jcp.oc - jcp.oc_without_padding);
        dw_bias = dw_padded_bias;
    }

    parallel(0, (size_t)work_amount, ker);

    if (pd()->wants_zero_pad_dst())
        output_memory_primitive(0)->zero_pad();
}

/* convolution backward wtr data */

void jit_avx2_1x1_convolution_bwd_data_t::execute_backward_data() const {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());

    auto rtus_space = scratchpad().get<data_t>(key_conv_rtus_space);

    const auto &jcp = kernel_->jcp;
    const int MB = pd()->MB();

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);
    const int ndims = diff_dst_d.ndims();

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];

    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;
    const int nb_oc_blocking = jcp.nb_reduce_blocking;

    const int work_amount = MB * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto ker = [&](const int ithr, const int nthr) {
        auto p = jit_1x1_conv_call_s();
        auto rp = rtus_driver_t<avx2>::call_params_t();

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        int load_step = 0;
        for (int icb = 0; icb < jcp.nb_load; icb += load_step) {
            load_step = step(jcp.nb_load_blocking, jcp.nb_load - icb,
                    jcp.nb_load_blocking_max);

            p.load_dim = this_block_size(icb * jcp.ic_block, jcp.ic,
                    load_step * jcp.ic_block);
            rp.icb = p.load_dim / jcp.ic_block;

            int bcast_step;
            for (int iwork = start; iwork < end; iwork += bcast_step) {
                int n{0}, g{0}, osb{0};
                nd_iterator_init(iwork, n, MB, g, jcp.ngroups, osb,
                        jcp.nb_bcast);

                bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                        jcp.nb_bcast_blocking_max);
                bcast_step = nstl::min(bcast_step, end - iwork);

                const int os = osb * os_block;
                p.bcast_dim = this_block_size(os, jcp.os,
                        bcast_step * os_block);
                rp.os = p.bcast_dim;

                const int oh = os / jcp.ow;
                const int ow = os % jcp.ow;
                const int ih = nstl::max(oh * stride_h - pad_t, 0);
                const int iw = nstl::max(ow * stride_w - pad_l, 0);
                rp.iw_start = iw;

                const int _icb = g * nb_ic + icb;
                rp.src = diff_src + data_blk_off(diff_src_d, n, _icb, ih, iw);
                if (pd()->rtus_.reduce_src_) {
                    rp.ws = rtus_space
                        + ithr * pd()->rtus_.space_per_thread_;
                    p.output_data = rp.ws;
                } else
                    p.output_data = rp.src;

                for (int ocb = 0; ocb < jcp.nb_reduce;
                        ocb += jcp.nb_reduce_blocking) {
                    const int _ocb = g * nb_oc + ocb;
                    size_t diff_dst_off = data_blk_off(diff_dst_d, n, _ocb, oh,
                        ow);
                    p.bcast_data = &diff_dst[diff_dst_off];

                    p.load_data = &weights[pd()->with_groups()
                        ? weights_d.blk_off(g, ocb, icb)
                        : weights_d.blk_off(ocb, icb)];

                    p.first_last_flag = ocb == 0 ? FLAG_REDUCE_FIRST : 0;

                    p.reduce_dim = this_block_size(ocb * jcp.oc_block, jcp.oc,
                            nb_oc_blocking * jcp.oc_block);

                    kernel_->jit_ker(&p);
                }

                if (pd()->rtus_.reduce_src_)
                    rtus_driver_->ker_(&rp);
            }
        }
    };

    parallel(0, (size_t)work_amount, ker);
}

/* convolution backward wtr weights */

jit_avx2_1x1_convolution_bwd_weights_t::jit_avx2_1x1_convolution_bwd_weights_t(
        const pd_t *apd, const input_vector &inputs,
        const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr)
    , rtus_driver_(nullptr)
{
    kernel_ = new jit_avx2_1x1_conv_kernel_f32(pd()->jcp_, jit_conv_conf_t(), *pd()->attr());
    reducer_weights_ =
        new cpu_reducer_2d_t<data_type::f32>(pd()->reducer_wei_conf_);
    reducer_bias_ = new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_);
    init_rtus_driver<avx2>(this);
}

void jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias_in = reinterpret_cast<data_t *>(this->memory(1));

    auto scratchpad = this->scratchpad();

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_pd(1));

    const auto &jcp = kernel_->jcp;
    auto rtus_space = scratchpad.get<data_t>(key_conv_rtus_space);

    data_t *diff_bias = pd()->wants_padded_bias()
        ? scratchpad.get<data_t>(key_conv_padded_bias) : diff_bias_in;

    auto reducer_bia_scratchpad = memory_tracking::grantor_t(scratchpad,
            prefix_reducer_bia);
    auto rb = this->reducer_bias_;
    rb->init(reducer_bia_scratchpad);

    auto reducer_wei_scratchpad = memory_tracking::grantor_t(scratchpad,
            prefix_reducer_wei);
    auto rw = this->reducer_weights_;
    rw->init(reducer_wei_scratchpad);

    const int ndims = diff_dst_d.ndims();
    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;
    const int bcast_work = div_up(nb_ic, nb_ic_blocking);

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;
    const int load_work = div_up(nb_oc, nb_oc_blocking);

    const int sp_dim = jcp.reduce_dim;
    const int mb_sp_work = jcp.mb * sp_dim;

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto oc_ic_sp_loop = [=](int sp_start, int sp_end, bool first_image,
            data_t *store_to, size_t store_to_ld, const data_t *diff_dst,
            const data_t *src, int ithr) {
        auto p = jit_1x1_conv_call_s();
        auto rp = rtus_driver_t<avx2>::call_params_t();

        p.output_stride = store_to_ld * sizeof(float);
        const int sp_step_def = jcp.nb_reduce_blocking * jcp.reduce_block;

        int oc_b_step = 0;
        for (int oc_b = 0; oc_b < nb_oc_blocking; oc_b += oc_b_step) {
            oc_b_step = step(12, nb_oc_blocking - oc_b, 18);
            p.load_dim = oc_b_step * jcp.oc_block;

            int ic_b_step = 0;
            for (int ic_b = 0; ic_b < nb_ic_blocking; ic_b += ic_b_step) {
                ic_b_step = step(12, nb_ic_blocking - ic_b, 18);
                p.bcast_dim = ic_b_step * jcp.ic_block;
                rp.icb = p.bcast_dim / jcp.ic_block;

                p.output_data = store_to + oc_b * store_to_ld
                    + ic_b * jcp.ic_block * jcp.oc_block;

                /* spatial reduction */
                int sp_step = 0;
                for (int sp = sp_start; sp < sp_end; sp += sp_step) {
                    sp_step = step(sp_step_def, sp_end - sp, 192);
                    p.reduce_dim = sp_step;
                    rp.os = p.reduce_dim;

                    p.first_last_flag = sp == sp_start && first_image
                        ? FLAG_REDUCE_FIRST : 0;

                    p.load_data = diff_dst
                        + (oc_b * jcp.reduce_dim + sp) * jcp.oc_block;

                    if (pd()->rtus_.reduce_src_) {
                        const int oh = sp / jcp.ow;
                        const int ow = sp % jcp.ow;

                        const int ih = nstl::max(oh * stride_h - pad_t, 0);
                        const int iw = nstl::max(ow * stride_w - pad_l, 0);
                        rp.iw_start = iw;

                        rp.ws = rtus_space
                            + ithr * pd()->rtus_.space_per_thread_
                            + (ic_b * jcp.is + sp) * jcp.ic_block;
                        if (ndims == 3)
                            rp.src = src
                                + iw * src_d.blocking_desc().strides[0][2];
                        else
                            rp.src = src
                                + ih * src_d.blocking_desc().strides[0][2]
                                + iw * src_d.blocking_desc().strides[0][3];

                        if (oc_b == 0)
                            rtus_driver_->ker_(&rp);

                        p.bcast_data = rp.ws;
                    } else
                        p.bcast_data = src
                            + (ic_b * jcp.reduce_dim + sp) * jcp.ic_block;

                    kernel_->jit_ker(&p);
                }
            }
        }
    };

    auto ker = [&](const int ithr, const int nthr) {
        assert(nthr == rw->balancer().nthr_);

        const int w_njobs = rw->balancer().ithr_njobs(ithr);
        if (w_njobs == 0) return;

        /* setup: independent work (oc, ic) */
        const int w_job_start = rw->balancer().ithr_job_off(ithr);
        int g{0}, load_i{0}, bcast_i{0};
        nd_iterator_init(w_job_start, g, jcp.ngroups, load_i, load_work,
                bcast_i, bcast_work);

        /* setup: reduction work (mb, sp) */
        int mb_sp_start{0}, mb_sp_end{0};
        balance211(mb_sp_work, rw->balancer().nthr_per_group_,
                rw->balancer().id_in_group(ithr), mb_sp_start, mb_sp_end);
        int img_start{0}, sp_start{0};
        nd_iterator_init(mb_sp_start, img_start, jcp.mb, sp_start, sp_dim);

        /* independent work */
        for (int iwork = 0; iwork < w_njobs; ++iwork) {
            const int oc_b = nb_oc_blocking * load_i;
            const int ic_b = nb_ic_blocking * bcast_i;

            const int _ic_b = g * nb_ic + ic_b;
            const int _oc_b = g * nb_oc + oc_b;

            data_t *store_to;
            size_t store_to_ld;

            if (rw->balancer().nthr_per_group_ == 1) {
                const size_t off = pd()->with_groups()
                    ? diff_weights_d.blk_off(g, oc_b, ic_b)
                    : diff_weights_d.blk_off(oc_b, ic_b);
                store_to = &diff_weights[off];
                store_to_ld = jcp.ic * jcp.oc_block;
            } else {
                const size_t off = iwork * rw->balancer().job_size_;
                store_to =
                    rw->get_local_ptr(ithr, reducer_wei_scratchpad) + off;
                store_to_ld = nb_ic_blocking * jcp.ic_block * jcp.oc_block;
            }

            /* reduction work */
            int img = img_start;
            int sp = sp_start;
            int sp_step = 0;
            for (int mb_sp = mb_sp_start; mb_sp < mb_sp_end; mb_sp += sp_step)
            {
                sp_step = nstl::min(sp_dim - sp, mb_sp_end - mb_sp);

                const bool first_image = img == img_start;
                oc_ic_sp_loop(sp, sp + sp_step, first_image, store_to,
                        store_to_ld, &diff_dst[diff_dst_d.blk_off(img, _oc_b)],
                        &src[src_d.blk_off(img, _ic_b)], ithr);

                sp = 0;
                img += 1;
            }

            nd_iterator_step(g, jcp.ngroups, load_i, load_work, bcast_i,
                             bcast_work);
        }
        rw->reduce(ithr, diff_weights, reducer_wei_scratchpad);
    };

    auto ker_bias = [&](int ithr, int nthr) {
        assert(nthr == rb->balancer().nthr_);

        const int b_job_start = rb->balancer().ithr_job_off(ithr);
        const int b_njobs = rb->balancer().ithr_njobs(ithr);

        if (b_njobs == 0) return;

        /* reduction dimension */
        int img_start{0}, img_end{0};
        balance211(jcp.mb, rb->balancer().nthr_per_group_,
                rb->balancer().id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start{0}, ocb_start{0};
        nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, nb_oc);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const size_t _oc = g * nb_oc + ocb;

                const data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                data_t *d_bias =
                    rb->get_local_ptr(ithr, diff_bias, reducer_bia_scratchpad)
                    + b_job_loc * rb->balancer().job_size_;

                if (img == img_start)
                    for (int o = 0; o < 8; ++o) d_bias[o] = 0.;

                for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
                    PRAGMA_OMP_SIMD()
                    for (int o = 0; o < 8; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += 8;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, nb_oc);
            }
        }
        rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    parallel(0, (size_t)mkldnn_get_max_threads(), [&](const int ithr, const int nthr) {
        ker(ithr, nthr);
        if (pd()->with_bias())
            ker_bias(ithr, nthr);
    });

    /* TODO: put this in ker_bias */
    if (pd()->wants_padded_bias()) {
        assert(jcp.ngroups == 1);
        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            diff_bias_in[oc] = diff_bias[oc];
    }
}

}
}
}
