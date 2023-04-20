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
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static void maxpool_w8_int8_nhwc(const int8_t *src, int8_t *dst, struct csinn_pool_params *params,
                                 int oh, int ow, int idx_h_start, int idx_h_end, int in_w,
                                 int out_w, int in_c)
{
    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;

    const int idx_w_start = -pad_left + ow * stride_w;
    const int idx_w_end = idx_w_start + kernel_w;

    vint8m1_t _max0, _max1, _max2, _max3;
    vint8m1_t _max4, _max5, _max6, _max7;

    int vl;
    int c = 0;
    while (c < in_c) {
        vl = vsetvl_e8m1(in_c - c);
        _max0 = vmv_v_x_i8m1(-128, vl);
        _max1 = _max2 = _max3 = _max4 = _max0;
        _max5 = _max6 = _max7 = _max0;

        for (int h = idx_h_start; h < idx_h_end; h++) {
            for (int w = idx_w_start; w < idx_w_end; w++) {
                const int8_t *in_ptr = src + (h * in_w + w) * in_c + c;
                _max0 = vmax_vv_i8m1(_max0, vle8_v_i8m1(in_ptr + 0 * stride_w * in_c, vl), vl);
                _max1 = vmax_vv_i8m1(_max1, vle8_v_i8m1(in_ptr + 1 * stride_w * in_c, vl), vl);
                _max2 = vmax_vv_i8m1(_max2, vle8_v_i8m1(in_ptr + 2 * stride_w * in_c, vl), vl);
                _max3 = vmax_vv_i8m1(_max3, vle8_v_i8m1(in_ptr + 3 * stride_w * in_c, vl), vl);
                _max4 = vmax_vv_i8m1(_max4, vle8_v_i8m1(in_ptr + 4 * stride_w * in_c, vl), vl);
                _max5 = vmax_vv_i8m1(_max5, vle8_v_i8m1(in_ptr + 5 * stride_w * in_c, vl), vl);
                _max6 = vmax_vv_i8m1(_max6, vle8_v_i8m1(in_ptr + 6 * stride_w * in_c, vl), vl);
                _max7 = vmax_vv_i8m1(_max7, vle8_v_i8m1(in_ptr + 7 * stride_w * in_c, vl), vl);
            }
        }
        int8_t *out_ptr = dst + (oh * out_w + ow) * in_c + c;
        vse8_v_i8m1(out_ptr + 0 * in_c, _max0, vl);
        vse8_v_i8m1(out_ptr + 1 * in_c, _max1, vl);
        vse8_v_i8m1(out_ptr + 2 * in_c, _max2, vl);
        vse8_v_i8m1(out_ptr + 3 * in_c, _max3, vl);
        vse8_v_i8m1(out_ptr + 4 * in_c, _max4, vl);
        vse8_v_i8m1(out_ptr + 5 * in_c, _max5, vl);
        vse8_v_i8m1(out_ptr + 6 * in_c, _max6, vl);
        vse8_v_i8m1(out_ptr + 7 * in_c, _max7, vl);
        c += vl;
    }
}

static void maxpool_border_int8_nhwc(const int8_t *src, int8_t *dst,
                                     struct csinn_pool_params *params, int oh, int ow,
                                     int idx_h_start, int idx_h_end, int in_w, int out_w, int in_c)
{
    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;

    int i_w_start = -pad_left + ow * stride_w;
    int i_w_end = i_w_start + kernel_w;
    const int idx_w_start = max(i_w_start, 0);
    const int idx_w_end = min(i_w_end, in_w);

    int vl;
    int c = 0;
    while (c < in_c) {
        vl = vsetvl_e8m1(in_c - c);
        vint8m1_t _max = vmv_v_x_i8m1(-128, vl);
        for (int h = idx_h_start; h < idx_h_end; h++) {
            for (int w = idx_w_start; w < idx_w_end; w++) {
                const int8_t *in_ptr = src + (h * in_w + w) * in_c + c;
                _max = vmax_vv_i8m1(_max, vle8_v_i8m1(in_ptr, vl), vl);
            }
        }
        int8_t *out_ptr = dst + (oh * out_w + ow) * in_c + c;
        vse8_v_i8m1(out_ptr, _max, vl);
        c += vl;
    }
}

int shl_rvv_maxpool_nhwc_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params)
{
    int8_t *input_data = (int8_t *)input->data;
    int8_t *output_data = (int8_t *)output->data;

    int batch = input->dim[0];
    int in_h = input->dim[1];
    int in_w = input->dim[2];
    int in_c = input->dim[3];

    int out_h = output->dim[1];
    int out_w = output->dim[2];
    int out_c = output->dim[3];

    int kernel_h = params->filter_height;
    int kernel_w = params->filter_width;
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int pad_left = params->pad_left;
    int pad_top = params->pad_top;

    int dst_1x8_start_w = max((pad_left + stride_w - 1) / stride_w, 0);
    int dst_1x8_end_w = min((in_w + pad_left - kernel_w) / stride_w + 1, out_w);

    for (int b = 0; b < batch; b++) {
        int8_t *in_ptr = input_data + b * in_h * in_w;
        int8_t *out_ptr = output_data + b * out_h * out_w;

        for (int oh = 0; oh < out_h; oh++) {
            int i_h_start = -pad_top + oh * stride_h;
            int i_h_end = i_h_start + kernel_h;
            const int idx_h_start = max(i_h_start, 0);
            const int idx_h_end = min(i_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                maxpool_border_int8_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c);
            }
            for (; ow + 8 <= dst_1x8_end_w; ow += 8) {
                maxpool_w8_int8_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                     out_w, in_c);
            }
            for (; ow < out_w; ow++) {
                maxpool_border_int8_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c);
            }
        }
    }
    return CSINN_TRUE;
}
