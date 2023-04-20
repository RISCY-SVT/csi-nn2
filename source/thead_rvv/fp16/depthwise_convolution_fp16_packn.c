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

#include "shl_thead_rvv.h"

/*************************************************************
 * note: VLEN = 128/256 ... flexible vlen
 *************************************************************/
int shl_rvv_dwconv_packn_fp16(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params)
{
    __fp16 *input_data = (__fp16 *)input->data;
    __fp16 *output_data = (__fp16 *)output->data;
    __fp16 *kernel_data = (__fp16 *)kernel->data;
    __fp16 *bias_data = (__fp16 *)bias->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];  // group = in_channel
    int in_h = input->dim[2];
    int in_w = input->dim[3];

    int out_c = output->dim[1];
    int out_h = output->dim[2];
    int out_w = output->dim[3];

    int kernel_h = kernel->dim[2];
    int kernel_w = kernel->dim[3];
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;

    const int packn = csrr_vlenb() / sizeof(__fp16);
    const int vl = vsetvl_e16m1(packn);

    __fp16 *input_padd_buf =
        (__fp16 *)shl_mem_alloc(in_c * padded_in_h * padded_in_w * sizeof(__fp16));

#pragma omp parallel for num_threads(1)
    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_fp16(input_data, input_padd_buf, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left);
        for (int c = 0; c + packn <= in_c; c += packn) {
            const __fp16 *src = input_padd_buf + c * padded_in_h * padded_in_w;
            __fp16 *dst = output_data + c * out_h * out_w;
            const __fp16 *kernel0 = kernel_data + c * kernel_h * kernel_w;
            vfloat16m1_t _bias0 =
                bias_data ? vle16_v_f16m1(bias_data + c, vl) : vfmv_v_f_f16m1(0.0f, vl);

            for (int oh = 0; oh < out_h; oh++) {
                int i_h_start = oh * stride_h;
                int i_h_end = i_h_start + kernel_h;
                for (int ow = 0; ow < out_w; ow++) {
                    int i_w_start = ow * stride_w;
                    int i_w_end = i_w_start + kernel_w;

                    vfloat16m1_t _acc = _bias0;
                    for (int ih = i_h_start, i_kh = 0; ih < i_h_end; ih++, i_kh++) {
                        for (int iw = i_w_start, i_kw = 0; iw < i_w_end; iw++, i_kw++) {
                            const __fp16 *in_ptr = src + (ih * padded_in_w + iw) * packn;
                            const __fp16 *k_ptr = kernel0 + (i_kh * kernel_w + i_kw) * packn;
                            vfloat16m1_t _rxx = vle16_v_f16m1(in_ptr, vl);
                            vfloat16m1_t _kxx = vle16_v_f16m1(k_ptr, vl);
                            _acc = vfmacc_vv_f16m1(_acc, _rxx, _kxx, vl);
                        }
                    }
                    __fp16 *out_ptr = dst + (oh * out_w + ow) * packn;
                    vse16_v_f16m1(out_ptr, _acc, vl);
                }
            }
        }
        input_data += in_c * in_h * in_w;
        output_data += out_c * out_h * out_w;
    }
    // requantize
    shl_rvv_sidcso_op_requantize_fp16(input, output, kernel);
    return CSINN_TRUE;
}
