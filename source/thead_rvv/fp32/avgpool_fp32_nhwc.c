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

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static void avgpool_w8_fp32_nhwc(const float *src, float *dst, struct csinn_pool_params *params,
                                 int oh, int ow, int idx_h_start, int idx_h_end, int in_w,
                                 int out_w, int in_c, enum avgpool_loc_enum loc)
{
    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;
    const int idx_w_start = -params->pad_left + ow * stride_w;
    const int idx_w_end = idx_w_start + kernel_w;

    int window_size = shl_rvv_avgpool_get_window_size(params, idx_h_start, idx_h_end, idx_w_start,
                                                      idx_w_end, loc);
    float ratio = 1.0f / window_size;

    vfloat32m1_t _acc0, _acc1, _acc2, _acc3;
    vfloat32m1_t _acc4, _acc5, _acc6, _acc7;

    int vl;
    int c = 0;
    while (c < in_c) {
        vl = vsetvl_e32m1(in_c - c);
        _acc0 = vfmv_v_f_f32m1(0.0f, vl);
        _acc1 = _acc2 = _acc3 = _acc0;
        _acc4 = _acc5 = _acc6 = _acc7 = _acc0;

        for (int h = idx_h_start; h < idx_h_end; h++) {
            for (int w = idx_w_start; w < idx_w_end; w++) {
                const float *in_ptr = src + (h * in_w + w) * in_c + c;
                _acc0 = vfadd_vv_f32m1(_acc0, vle32_v_f32m1(in_ptr + 0 * stride_w * in_c, vl), vl);
                _acc1 = vfadd_vv_f32m1(_acc1, vle32_v_f32m1(in_ptr + 1 * stride_w * in_c, vl), vl);
                _acc2 = vfadd_vv_f32m1(_acc2, vle32_v_f32m1(in_ptr + 2 * stride_w * in_c, vl), vl);
                _acc3 = vfadd_vv_f32m1(_acc3, vle32_v_f32m1(in_ptr + 3 * stride_w * in_c, vl), vl);
                _acc4 = vfadd_vv_f32m1(_acc4, vle32_v_f32m1(in_ptr + 4 * stride_w * in_c, vl), vl);
                _acc5 = vfadd_vv_f32m1(_acc5, vle32_v_f32m1(in_ptr + 5 * stride_w * in_c, vl), vl);
                _acc6 = vfadd_vv_f32m1(_acc6, vle32_v_f32m1(in_ptr + 6 * stride_w * in_c, vl), vl);
                _acc7 = vfadd_vv_f32m1(_acc7, vle32_v_f32m1(in_ptr + 7 * stride_w * in_c, vl), vl);
            }
        }

        float *out_ptr = dst + (oh * out_w + ow) * in_c + c;
        vse32_v_f32m1(out_ptr + 0 * in_c, vfmul_vf_f32m1(_acc0, ratio, vl), vl);
        vse32_v_f32m1(out_ptr + 1 * in_c, vfmul_vf_f32m1(_acc1, ratio, vl), vl);
        vse32_v_f32m1(out_ptr + 2 * in_c, vfmul_vf_f32m1(_acc2, ratio, vl), vl);
        vse32_v_f32m1(out_ptr + 3 * in_c, vfmul_vf_f32m1(_acc3, ratio, vl), vl);
        vse32_v_f32m1(out_ptr + 4 * in_c, vfmul_vf_f32m1(_acc4, ratio, vl), vl);
        vse32_v_f32m1(out_ptr + 5 * in_c, vfmul_vf_f32m1(_acc5, ratio, vl), vl);
        vse32_v_f32m1(out_ptr + 6 * in_c, vfmul_vf_f32m1(_acc6, ratio, vl), vl);
        vse32_v_f32m1(out_ptr + 7 * in_c, vfmul_vf_f32m1(_acc7, ratio, vl), vl);

        c += vl;
    }
}

static void avgpool_border_fp32_nhwc(const float *src, float *dst, struct csinn_pool_params *params,
                                     int oh, int ow, int idx_h_start, int idx_h_end, int in_w,
                                     int out_w, int in_c, enum avgpool_loc_enum loc)
{
    int kernel_w = params->filter_width;
    int stride_w = params->stride_width;

    int i_w_start = -params->pad_left + ow * stride_w;
    int i_w_end = i_w_start + kernel_w;
    const int idx_w_start = max(i_w_start, 0);
    const int idx_w_end = min(i_w_end, in_w);

    int window_size = shl_rvv_avgpool_get_window_size(params, idx_h_start, idx_h_end, idx_w_start,
                                                      idx_w_end, loc);
    float ratio = 1.0f / window_size;

    int vl;
    int c = 0;
    while (c < in_c) {
        vl = vsetvl_e32m1(in_c - c);
        vfloat32m1_t _acc = vfmv_v_f_f32m1(0.0f, vl);
        for (int h = idx_h_start; h < idx_h_end; h++) {
            for (int w = idx_w_start; w < idx_w_end; w++) {
                const float *in_ptr = src + (h * in_w + w) * in_c + c;
                _acc = vfadd_vv_f32m1(_acc, vle32_v_f32m1(in_ptr, vl), vl);
            }
        }
        float *out_ptr = dst + (oh * out_w + ow) * in_c + c;
        vse32_v_f32m1(out_ptr, vfmul_vf_f32m1(_acc, ratio, vl), vl);
        c += vl;
    }
}

int shl_rvv_avgpool_nhwc_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_pool_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

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

    int pad_top = params->pad_top;
    int pad_left = params->pad_left;

    int dst_start_h = max((pad_top + stride_h - 1) / stride_h, 0);
    int dst_end_h = min((in_h + pad_top - kernel_h) / stride_h + 1, out_h);
    int dst_1x8_start_w = max((pad_left + stride_w - 1) / stride_w, 0);
    int dst_1x8_end_w = min((in_w + pad_left - kernel_w) / stride_w + 1, out_w);

    for (int b = 0; b < batch; b++) {
        const float *in_ptr = input_data + b * in_h * in_w * in_c;
        float *out_ptr = output_data + b * out_h * out_w * out_c;

        int oh = 0;
        for (; oh < dst_start_h; oh++) {
            int in_h_start = -pad_top + oh * stride_h;
            int in_h_end = in_h_start + kernel_h;
            const int idx_h_start = max(in_h_start, 0);
            const int idx_h_end = min(in_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                avgpool_border_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c, AVGPOOL_LEFT_TOP);
            }
            for (; ow + 7 < dst_1x8_end_w; ow += 8) {
                avgpool_w8_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                     out_w, in_c, AVGPOOL_TOP);
            }
            for (; ow < out_w; ow++) {
                avgpool_border_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c, AVGPOOL_RIGHT_TOP);
            }
        }
        for (; oh < dst_end_h; oh++) {
            int in_h_start = -pad_top + oh * stride_h;
            int in_h_end = in_h_start + kernel_h;
            const int idx_h_start = max(in_h_start, 0);
            const int idx_h_end = min(in_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                avgpool_border_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c, AVGPOOL_LEFT);
            }
            for (; ow + 7 < dst_1x8_end_w; ow += 8) {
                avgpool_w8_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                     out_w, in_c, AVGPOOL_CENTER);
            }
            for (; ow < out_w; ow++) {
                avgpool_border_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c, AVGPOOL_RIGHT);
            }
        }
        for (; oh < out_h; oh++) {
            int in_h_start = -pad_top + oh * stride_h;
            int in_h_end = in_h_start + kernel_h;
            const int idx_h_start = max(in_h_start, 0);
            const int idx_h_end = min(in_h_end, in_h);
            int ow = 0;
            for (; ow < dst_1x8_start_w; ow++) {
                avgpool_border_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c, AVGPOOL_LEFT_BOTTOM);
            }
            for (; ow + 7 < dst_1x8_end_w; ow += 8) {
                avgpool_w8_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end, in_w,
                                     out_w, in_c, AVGPOOL_BOTTOM);
            }
            for (; ow < out_w; ow++) {
                avgpool_border_fp32_nhwc(in_ptr, out_ptr, params, oh, ow, idx_h_start, idx_h_end,
                                         in_w, out_w, in_c, AVGPOOL_RIGHT_BOTTOM);
            }
        }
    }
    return CSINN_TRUE;
}
