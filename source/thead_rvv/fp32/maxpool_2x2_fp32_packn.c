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
 * note: support flexible vlen
 *************************************************************/

/*
    TODO: 所有的 kernel_size 和 stride 的都可以写成一个接口，库大小被优化了， 可以参考:
   /lhome/shaowg/hhb_workspace/csinn2/source/i805_ref/pooling/shl_pool_q7_HWC.c
   或者参考 ppl.nn 中 maxpool
*/

int shl_rvv_maxpool2x2s2_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                    struct csinn_pool_params *params)
{
    // 1. 统一padding之后再计算，不考虑padiing
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    int batch = input->dim[0];
    int in_c = input->dim[1];
    int in_h = input->dim[2];
    int in_w = input->dim[3];
    int input_size = in_c * in_h * in_w;

    int out_h = output->dim[2];
    int out_w = output->dim[3];
    int output_size = in_c * out_h * out_w;

    int padded_in_h = in_h + params->pad_top + params->pad_down;
    int padded_in_w = in_w + params->pad_left + params->pad_right;
    int padded_in_hw = padded_in_w * padded_in_h;

    const int packn = csrr_vlenb() / sizeof(float);
    const int vl = vsetvl_e32m1(packn);

    float *input_ncxhwx = (float *)shl_mem_alloc(in_c * padded_in_hw * sizeof(float));
    int tailstep = (padded_in_w - 2 * out_w + padded_in_w) * packn;

    for (int b = 0; b < batch; b++) {
        shl_rvv_pad_input_packn_fp32(input_data, input_ncxhwx, in_c, in_h, in_w, padded_in_h,
                                     padded_in_w, params->pad_top, params->pad_left);

        for (int c = 0; c + packn - 1 < in_c; c += packn) {
            float *out0 = output_data + c * out_h * out_w;
            const float *line0 = input_ncxhwx + c * padded_in_h * padded_in_w;
            const float *line1 = line0 + padded_in_w * packn;

            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    vfloat32m1_t _max = vle32_v_f32m1(line0, vl);
                    _max = vfmax_vv_f32m1(_max, vle32_v_f32m1(line0 + packn, vl), vl);
                    _max = vfmax_vv_f32m1(_max, vle32_v_f32m1(line1, vl), vl);
                    _max = vfmax_vv_f32m1(_max, vle32_v_f32m1(line1 + packn, vl), vl);
                    vse32_v_f32m1(out0, _max, vl);

                    line0 += packn * 2;
                    line1 += packn * 2;
                    out0 += packn;
                }
                line0 += tailstep;
                line1 += tailstep;
            }
        }
        input_data += input_size;
        output_data += output_size;
    }
    shl_mem_free(input_ncxhwx);
    return CSINN_TRUE;
}
