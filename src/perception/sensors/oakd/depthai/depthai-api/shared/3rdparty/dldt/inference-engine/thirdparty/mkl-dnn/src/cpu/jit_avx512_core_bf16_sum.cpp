/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include <float.h>
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "bfloat16_utils.hpp"
#include "jit_avx512_core_bf16_sum.hpp"

#define GET_OFF(field) offsetof(jit_sum_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::cpu::bf16_cvt_utils;

using namespace Xbyak;
void jit_avx512_core_bf16_sum_kernel::loop_iteration(int current_unroll)
{
    Label loop_label, exit_label;
    const int num_compute_elements = 2 * f32_simd_w_ * current_unroll;
    size_t src_shift = 2 * f32_simd_w_ * jsp.typesize_in;
    size_t dst_shift = f32_simd_w_* jsp.typesize_out;

    L(loop_label);
    cmp(reg_sz, num_compute_elements);
    jl(exit_label, T_NEAR);
    for (int u_idx = 0; u_idx < current_unroll; u_idx++) {
        zmm_t vacc0 = Zmm(acc_vreg_idx(u_idx, 0));
        zmm_t vacc1 = Zmm(acc_vreg_idx(u_idx, 1));
        vpxord(vacc0, vacc0, vacc0);
        vpxord(vacc1, vacc1, vacc1);

        int num_acc_iters = utils::div_up(jsp.num_srcs, 2);
        for (int acc_iter = 0; acc_iter < num_acc_iters; acc_iter++) {
            int isrc0 = 2 * acc_iter;
            int isrc1 = 2 * acc_iter + 1;
            zmm_t vscale = Zmm(scale_vreg_idx(acc_iter));
            zmm_t vsrc0 = Zmm(src_vreg_idx(u_idx, isrc0));
            zmm_t vsrc1 = Zmm(src_vreg_idx(u_idx, isrc1));
            zmm_t vtmp = Zmm(tmp_vreg_idx(u_idx, acc_iter));
            vmovups(vsrc0, zword[reg_src[isrc0] + u_idx * src_shift]);
            if (num_acc_iters * 2 > jsp.num_srcs
                && acc_iter == num_acc_iters - 1)
                vpxord(vtmp, vtmp, vtmp); /* imitate additional zero input
                                             if number of srcs is odd */
            else
                vmovups(vtmp, zword[reg_src[isrc1] + u_idx * src_shift]);
            vshuff64x2(vsrc1, vsrc0, vtmp, 0xEE);
            vpermw(vsrc1, zmm_idx, vsrc1);
            vshuff64x2(vsrc0, vsrc0, vtmp, 0x44);
            vpermw(vsrc0, zmm_idx, vsrc0);

                if (!jsp.is_cpx) {
                    bf16_emu_->r_vdpbf16ps(vacc0, vsrc0, vscale);
                    vpbroadcastd(vscale,
                        ptr[reg_scales + 2 * acc_iter * jsp.typesize_in]);
                    bf16_emu_->r_vdpbf16ps(vacc1, vsrc1, vscale);
                } else {
                    vdpbf16ps(vacc0, vsrc0, vscale);
                    vdpbf16ps(vacc1, vsrc1, vscale);
                }
        }

        if (!jsp.is_bf16_dst) {
            vmovups(zword[reg_dst + 2 * u_idx * dst_shift], vacc0);
            vmovups(zword[reg_dst + (2 * u_idx + 1) * dst_shift], vacc1);
        } else {
            if (jsp.is_cpx) {
                zmm_t zmm_str = Zmm(tmp_vreg_idx(u_idx, 0));
                vcvtne2ps2bf16(zmm_str, vacc1, vacc0);
                vmovups(zword[reg_dst + 2 * u_idx * dst_shift], zmm_str);
            } else {
                auto ymm_str = Ymm(tmp_vreg_idx(u_idx, 0));
                bf16_emu_->r_vcvtneps2bf16(ymm_str, vacc0);
                vmovups(yword[reg_dst + 2 * u_idx * dst_shift], ymm_str);
                bf16_emu_->r_vcvtneps2bf16(ymm_str, vacc1);
                vmovups(yword[reg_dst + (2 * u_idx + 1) * dst_shift], ymm_str);
            }
        }
    }
    sub(reg_sz, num_compute_elements);
    for (int s = 0; s < jsp.num_srcs; s++)
        add(reg_src[s], current_unroll * src_shift);
    add(reg_dst, 2 * current_unroll * dst_shift);
    jge(loop_label, T_NEAR);

    L(exit_label);
}

void jit_avx512_core_bf16_sum_kernel::generate()
{
    preamble();

    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_srcs, ptr[param + GET_OFF(srcs)]);

    for (int s = 0; s < jsp.num_srcs; s++)
         mov(reg_src[s], ptr[reg_srcs + sizeof(void*) * s]);

    mov(reg_scales, ptr[param + GET_OFF(scales)]);
    mov(reg_sz, ptr[param + GET_OFF(size)]);

    Label tail_label, exit_label, mask_label;

    mov(reg_idx_table, idx_table);
    vmovups(zmm_idx, ptr[reg_idx_table]);

    int num_acc_iters = utils::div_up(jsp.num_srcs, 2);
    for (int acc_iter = 0; acc_iter < num_acc_iters; acc_iter++) {
        zmm_t vscale = Zmm(scale_vreg_idx(acc_iter));
        vpbroadcastd(vscale, ptr[reg_scales + 2 * acc_iter * jsp.typesize_in]);
    }

    if (!jsp.is_cpx) bf16_emu_->init_vcvtneps2bf16();
    if (jsp.loop_unroll > 1)
        loop_iteration(jsp.loop_unroll);

    loop_iteration(1);

    // tail processing
    L(tail_label);
    cmp(reg_sz, 0);
    jle(exit_label, T_NEAR);

    const int bf16_half_reg = f32_simd_w_;
    mov(reg32_mask, 0xffff);
    cmp(reg_sz, bf16_half_reg);
    jge(mask_label, T_NEAR);

    mov(reg32_mask, 1);
    mov(rcx, reg_sz);
    shl(reg32_mask, cl);
    sub(reg32_mask, 1);

    L(mask_label);
    kmovd(k_mask, reg32_mask);
    zmm_t vacc = Zmm(acc_vreg_idx(0, 0));
    vpxord(vacc, vacc, vacc);

    for (int acc_iter = 0; acc_iter < num_acc_iters; acc_iter++) {
        int isrc0 = 2 * acc_iter;
        int isrc1 = 2 * acc_iter + 1;
        zmm_t vscale = Zmm(scale_vreg_idx(acc_iter));
        zmm_t vsrc = Zmm(src_vreg_idx(0, isrc0));
        ymm_t vysrc0 = Ymm(src_vreg_idx(0, isrc0));
        ymm_t vysrc1 = Ymm(src_vreg_idx(0, isrc1));
        vpxord(vysrc0, vysrc0, vysrc0);
        vpxord(vysrc1, vysrc1, vysrc1);

        vmovdqu16(vysrc0 | k_mask | T_z, yword[reg_src[isrc0]]);
        if (!(num_acc_iters * 2 > jsp.num_srcs
                && acc_iter == num_acc_iters - 1))
            vmovdqu16(vysrc1 | k_mask | T_z, yword[reg_src[isrc1]]);
        vinserti64x4(vsrc, vsrc, vysrc1, 0x1);
        vpermw(vsrc, zmm_idx, vsrc);

        if (!jsp.is_cpx) {
            bf16_emu_->r_vdpbf16ps(vacc, vsrc, vscale);
        } else {
            vdpbf16ps(vacc, vsrc, vscale);
        }
    }
    if (!jsp.is_bf16_dst) {
        vmovups(zword[reg_dst] | k_mask, vacc);
    } else {
        if (jsp.is_cpx) {
            auto ymm_str = Ymm(tmp_vreg_idx(0, 0));
            vcvtneps2bf16(ymm_str, vacc);
            vmovdqu16(yword[reg_dst] | k_mask, ymm_str);
        } else {
            auto ymm_str = Ymm(tmp_vreg_idx(0, 0));
            bf16_emu_->r_vcvtneps2bf16(ymm_str, vacc);
            vmovdqu16(yword[reg_dst] | k_mask, ymm_str);
        }
    }

    sub(reg_sz, bf16_half_reg);
    cmp(reg_sz, 0);
    jle(exit_label, T_NEAR);

    for (int s = 0; s < jsp.num_srcs; s++)
        add(reg_src[s], bf16_half_reg * jsp.typesize_in);
    add(reg_dst, f32_simd_w_ * jsp.typesize_out);

    jmp(tail_label, T_NEAR);

    L(exit_label);
    postamble();

    align(64);
    L(idx_table);
    const uint16_t _idx[] = { 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23,8,24,
                              9,25,10,26,11,27,12,28,13,29,14,30,15,31 };
    for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
        dw(_idx[i]);
}

status_t jit_avx512_core_bf16_sum_kernel::init_conf(
                                jit_sum_conf_t &jsp,
                                const int num_srcs,
                                const cpu_memory_pd_t &dst_d)
{
    jsp.is_cpx = mayiuse(avx512_core_bf16);

    jsp.num_srcs = num_srcs;
    jsp.loop_unroll = 0;

    const int max_unroll = 6; // maximum possible value of unroll is 6
    for (/*continue*/; jsp.loop_unroll < max_unroll; jsp.loop_unroll++)
    {
        int num_regs = num_vregs_required(jsp.loop_unroll + 1, jsp.num_srcs);
        if (num_regs > max_vregs_available(jsp.is_cpx))
            break;
    }
    const int bf16_simd_w = 32;
    if (jsp.loop_unroll == 0) return status::unimplemented;
    jsp.size_blocking = bf16_simd_w * jsp.loop_unroll;

    const memory_desc_wrapper o_d(&dst_d);
    jsp.is_bf16_dst = data_type::bf16 == o_d.data_type();

    jsp.typesize_in = sizeof(mkldnn_bfloat16_t);
    jsp.typesize_out = jsp.is_bf16_dst
                     ? sizeof(mkldnn_bfloat16_t)
                     : sizeof(float);

    return status::success;
}

template <data_type_t src_data_type, data_type_t dst_data_type>
void jit_bf16_sum_t<src_data_type, dst_data_type>::execute() const
{
    auto output = reinterpret_cast<dst_data_t *>(this->memory());
    const int num_arrs = pd()->n_inputs();
    const memory_desc_wrapper o_d(pd()->dst_pd());
    output += o_d.blk_off(0);
    const size_t nelems = o_d.nelems();
    const src_data_t *input_ptrs[max_num_arrs];
    /* Number of scales needs to be multiple of 2 in order
    to use VNNI instructions */
    src_data_t scales[max_num_arrs];
    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(pd()->src_pd(a));

        input_ptrs[a] = reinterpret_cast<const src_data_t *>(
                this->input_memory(a)) + i_d.blk_off(0);
    }
    cvt_float_to_bfloat16(scales, &pd()->scales_[0], num_arrs);
    if (num_arrs % 2 != 0)
        scales[num_arrs] = cvt_float_to_bfloat16(0.0f);

    const size_t half_L1 = 16 * 1024; // bytes
    const size_t num_elems_in_block = utils::rnd_up(
                 utils::div_up(half_L1,
                     num_arrs * sizeof(src_data_t) + sizeof(dst_data_t)),
                 pd()->jsp_.size_blocking);
    const size_t num_blocks = nelems / num_elems_in_block;
    const size_t tail = nelems % num_elems_in_block;

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(num_blocks, nthr, ithr, start, end);
        auto arg = jit_sum_call_s();
        const src_data_t *local_input_ptrs[max_num_arrs];
        dst_data_t *local_output;

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * num_elems_in_block;
            for (int a = 0; a < num_arrs; ++a) {
                local_input_ptrs[a] = &input_ptrs[a][start_e];
            }
            local_output = &output[start_e];
            arg.srcs = (const void **)local_input_ptrs;
            arg.dst = (const void *)local_output;
            arg.scales = (const void *)scales;
            arg.size = num_elems_in_block;
            kernel_->jit_ker(&arg);
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            for (int a = 0; a < num_arrs; ++a) {
                local_input_ptrs[a] = &input_ptrs[a][start_e];
            }
            local_output = &output[start_e];
            arg.srcs = (const void **)local_input_ptrs;
            arg.dst = (const void *)local_output;
            arg.scales = (const void *)scales;
            arg.size = tail;
            kernel_->jit_ker(&arg);
        }
    });
}

template struct jit_bf16_sum_t<data_type::bf16, data_type::f32>;
template struct jit_bf16_sum_t<data_type::bf16, data_type::bf16>;

}
}
}
