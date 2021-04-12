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

#include "convolution_kernel_selector.h"
#include "convolution_kernel_ref.h"
#include "convolution_kernel_bfyx_1x1_opt.h"
#include "convolution_kernel_bfyx_gemm_like.h"
#include "convolution_kernel_bfyx_direct_10_12_16.h"
#include "convolution_kernel_bfyx_os_iyx_osv16.h"
#include "convolution_kernel_bfyx_os_iyx_osv16_2_sg.h"
#include "convolution_kernel_yxfb_ref.h"
#include "convolution_kernel_yxfb_yxio_b16.h"
#include "convolution_kernel_yxfb_yxio_b8.h"
#include "convolution_kernel_yxfb_yxio_b1_block.h"
#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"
// #include "convolution_kernel_bfyx_3x3_dw_opt.h"
#include "convolution_kernel_winograd_2x3_s1.h"
#include "convolution_kernel_bfyx_1x1.h"
#include "convolution_kernel_bfyx_1x1_gemm_buf.h"
#include "convolution_kernel_winograd_2x3_s1_fused.h"
#include "convolution_kernel_winograd_6x3_s1_fused.h"
#include "convolution_kernel_mmad.h"
#include "convolution_kernel_mmad_blocks.h"
#include "convolution_kernel_1x1_gemm_mmad.h"
#include "convolution_kernel_byxf_af32_depthwise.h"
#include "convolution_kernel_mmad_batched.h"
#include "convolution_kernel_bfyx_depthwise_weights_lwg.h"
#include "convolution_kernel_mmad_slm_2x14_rep4.h"
#include "convolution_kernel_mmad_slm_7x7_rep4.h"
#include "convolution_kernel_byxf_fs_bs_yx_bsv4_fsv32.h"
#include "convolution_kernel_mmad_batched_block.h"
#include "convolution_kernel_mmad_batched_block_1x1.h"
#include "convolution_kernel_mmad_32x32sg_128x128wg_slm_int8.h"
#include "convolution_kernel_mmad_32x32sg_224x128wg_slm_int8.h"
#include "convolution_kernel_mmad_32x32sg_slm_int8.h"
#include "convolution_kernel_byx8_f4__fs_bs_yx_bsv4_fsv32.h"
#include "convolution_kernel_imad.h"
#include "convolution_kernel_fs_byx_fsv32.h"
#include "convolution_kernel_fs_byx_fsv32_1x1.h"
#include "convolution_kernel_bfyx_to_fs_byx_fsv32.h"
#include "convolution_kernel_fs_byx_fsv32_depthwise.h"
#include "convolution_kernel_bfyx_f16_depthwise.h"
#include "convolution_kernel_bfyx_f16_1x1.h"
#include "convolution_kernel_bfyx_f16.h"
#include "convolution_kernel_bfyx_to_bfyx_f16.h"
#include "deformable_convolution_kernel_bfyx_ref.h"
#include "deformable_convolution_kernel_bfyx_conv.h"
#include "deformable_convolution_kernel_bfyx_interp.h"
#include "convolution_kernel_bfzyx_f16_fp32.h"
#include "convolution_kernel_bfzyx_f16_fp16.h"
#include "convolution_kernel_af32_imad_1x1.h"
#include "convolution_kernel_b_fs_yx_fsv4_1x1.h"
#include "convolution_kernel_mmad_bfyx_to_b_fs_yx_fsv4.h"
#include "convolution_kernel_mmad_b_fs_yx_fsv32.h"
#include "convolution_kernel_mmad_b_fs_yx_fsv32_dw.h"
#include "convolution_kernel_mmad_bfyx_b_fs_yx_fsv32.h"

namespace kernel_selector {
convolution_kernel_selector::convolution_kernel_selector() {
    Attach<ConvolutionKernel_Ref>();
    Attach<DeformableConvolutionKernel_bfyx_Ref>();

    // bfyx_f16
    Attach<ConvolutionKernel_bfyx_f16_depthwise>();
    Attach<ConvolutionKernel_bfyx_f16_1x1>();
    Attach<ConvolutionKernel_bfyx_f16>();
    Attach<ConvolutionKernel_bfyx_to_bfyx_f16>();

    // fs_byx_fsv32
    Attach<ConvolutionKernel_fs_byx_fsv32>();
    Attach<ConvolutionKernel_fs_byx_fsv32_1x1>();
    Attach<ConvolutionKernel_fs_byx_fsv32_depthwise>();
    Attach<ConvolutionKernel_bfyx_to_fs_byx_fsv32>();

    // bfyx fp
    Attach<convolution_kernel_bfyx_1x1_opt>();
    Attach<ConvolutionKernel_bfyx_GEMMLike>();
    Attach<ConvolutionKernel_bfyx_Direct_10_10_12>();
    Attach<ConvolutionKernel_bfyx_os_iyx_osv16>();
    Attach<ConvolutionKernel_bfyx_1x1>();
    Attach<ConvolutionKernel_bfyx_1x1_gemm_buf>();
    Attach<ConvolutionKernel_bfyx_depthwise_weights_lwg>();
    // commented out to not get in our way, will enable in future after autotuning
    // Attach<ConvolutionKernel_bfyx_os_iyx_osv16_2_sg>();

    // yxfb fp
    Attach<ConvolutionKernel_yxfb_Ref>();
    Attach<ConvolutionKernel_yxfb_yxio_b16>();
    Attach<ConvolutionKernel_yxfb_yxio_b8>();
    Attach<ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x>();
    // Attach<ConvolutionKernel_yxfb_yxio_b1_block>(); // TODO: need to finish integration
    // Attach<ConvolutionKernel_bfyx_3x3_dw_opt>();

    // Winograd
    Attach<ConvolutionKernel_Winograd_2x3_s1>();
    Attach<ConvolutionKernel_Winograd_2x3_s1_fused>();
    Attach<ConvolutionKernel_Winograd_6x3_s1_fused>();

    // byxf_af32 int8
    Attach<ConvolutionKernel_MMAD>();
    Attach<ConvolutionKernel_MMAD_blocks>();
    Attach<ConvolutionKernel_af32_imad_1x1>();
    Attach<ConvolutionKernel_byxf_af32_depthiwise>();
    Attach<ConvolutionKernel_1x1_gemm_MMAD>();

    // fs_bs_yx_bsv4_fsv32 int8
    Attach<ConvolutionKernel_mmad_batched>();
    Attach<ConvolutionKernel_mmad_slm_2x14_rep4>();
    Attach<ConvolutionKernel_mmad_slm_7x7_rep4>();
    Attach<ConvolutionKernel_mmad_32x32sg_128x128wg_slm_int8>();
    Attach<ConvolutionKernel_mmad_32x32sg_224x128wg_slm_int8>();
    Attach<ConvolutionKernel_byxf_fs_bs_yx_bsv4_fsv32>();
    Attach<ConvolutionKernel_byx8_f4__fs_bs_yx_bsv4_fsv32>();
    Attach<ConvolutionKernel_mmad_batched_block>();
    Attach<ConvolutionKernel_mmad_batched_block_1x1>();
    // Attach<ConvolutionKernel_mmad_32x32sg_slm_int8>();

    // b_fs_yx_fsv4 kernels
    Attach<ConvolutionKernel_imad>();
    Attach<ConvolutionKernel_b_fs_yx_fsv4_1x1>();
    Attach<ConvolutionKernel_MMAD_bfyx_to_b_fs_yx_fsv4>();

    // b_fs_yx_fsv32 kernels
    Attach<ConvolutionKernel_MMAD_b_fs_yx_fsv32>();
    Attach<ConvolutionKernel_MMAD_b_fs_yx_fsv32_dw>();
    Attach<ConvolutionKernel_MMAD_bfyx_b_fs_yx_fsv32>();

    // 3D optimized
    Attach<ConvolutionKernel_bfzyx_f16_fp32>();
    Attach<ConvolutionKernel_bfzyx_f16_fp16>();
}

KernelsData convolution_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::CONVOLUTION);
}

deformable_conv_kernel_selector::deformable_conv_kernel_selector() {
    Attach<DeformableConvolutionKernel_bfyx_conv>();
}

KernelsData deformable_conv_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::CONVOLUTION);
}

deformable_interp_kernel_selector::deformable_interp_kernel_selector() {
    Attach<DeformableConvolutionKernel_bfyx_interp>();
}

KernelsData deformable_interp_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetAutoTuneBestKernel(params, options, KernelType::CONVOLUTION);
}


}  // namespace kernel_selector
