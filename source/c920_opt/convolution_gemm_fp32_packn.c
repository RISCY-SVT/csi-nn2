/*
 * Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* SHL version 2.1.x */

#include "shl_c920.h"

int shl_c920_conv_im2col_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];
    int32_t in_c = input->dim[1];
    int32_t in_h = input->dim[2];
    int32_t in_w = input->dim[3];
    int32_t out_c = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];
    int32_t ksize_h = kernel->dim[2];
    int32_t ksize_w = kernel->dim[3];
    int32_t stride_h = params->stride_height;
    int32_t stride_w = params->stride_width;

    int32_t m = out_c / group;
    int32_t in_cp = in_c / group;
    int32_t maxk = ksize_h * ksize_w;
    int32_t n = out_h * out_w;

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            // padding
            int padded_in_hw = (in_h + params->pad_top + params->pad_down) *
                               (in_w + params->pad_left + params->pad_right);
            float *input_pad_buf = (float *)shl_mem_alloc(in_cp * padded_in_hw * sizeof(float));
            shl_rvv_pad_input_packn_fp32(input_data, input_pad_buf, in_cp, in_h, in_w,
                                         (in_h + params->pad_top + params->pad_down),
                                         (in_w + params->pad_left + params->pad_right),
                                         params->pad_top, params->pad_left);

            // im2col
            const int packn = csrr_vlenb() / sizeof(float);
            const int vl = vsetvl_e32m1(packn);

            // [in_c/packn, maxk, out_h, out_w, packn]
            float *im2col_buf = (float *)shl_mem_alloc(in_cp / packn * maxk * out_h * out_w *
                                                       packn * sizeof(float));
            const int tailstep =
                ((in_w + params->pad_left + params->pad_right) * stride_h - out_w * stride_w) *
                packn;

            for (int c = 0; c + packn - 1 < in_cp; c += packn) {
                const float *img0 = input_pad_buf + c * padded_in_hw;
                float *dst_ptr = im2col_buf + c * maxk * out_h * out_w;

                for (int a = 0; a < ksize_h; a++) {
                    for (int b = 0; b < ksize_w; b++) {
                        const float *img1 =
                            img0 + a * (in_w + params->pad_left + params->pad_right) * packn +
                            b * packn;

                        for (int p = 0; p < out_h; p++) {
                            for (int q = 0; q < out_w; q++) {
                                vfloat32m1_t _tmp = vle32_v_f32m1(img1, vl);
                                img1 += stride_w * packn;
                                vse32_v_f32m1(dst_ptr, _tmp, vl);
                                dst_ptr += packn;
                            }
                            img1 += tailstep;
                        }
                    }
                }
            }
            shl_mem_free(input_pad_buf);

            // reorder(pack)
            float *reorder_buf = (float *)shl_mem_alloc(in_cp * maxk * n * sizeof(float));
            shl_rvv_reorder_input_z8_packn_fp32(im2col_buf, reorder_buf, in_cp * maxk, n, n);
            shl_mem_free(im2col_buf);

            // gemm
            float *ker_ptr = kernel_data + g * m * maxk * in_cp;
            float *bias_ptr = bias_data ? (bias_data + g * m) : NULL;
            shl_c920_ncxhwx_gemm_8xpack2n_fp32(output_data, ker_ptr, reorder_buf, bias_ptr, m,
                                               in_cp * maxk, n, n);
            shl_mem_free(reorder_buf);

            input_data += in_cp * in_h * in_w;
            output_data += m * n;
        }
    }
    return CSINN_TRUE;
}
