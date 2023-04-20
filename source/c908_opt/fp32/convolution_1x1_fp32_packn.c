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

#include "shl_c908.h"

void shl_c908_conv1x1s1_gemm_reorder_kernel_packn_fp32(struct csinn_tensor *kernel,
                                                       struct csinn_conv2d_params *params)
{
    shl_c908_conv_im2col_gemm_reorder_kernel_packn_fp32(kernel, params);
}

int shl_c908_conv1x1s1_gemm_packn_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                       struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                       struct csinn_conv2d_params *params)
{
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *kernel_data = (float *)kernel->data;
    float *bias_data = (float *)bias->data;

    int32_t group = params->group;
    int32_t batch = input->dim[0];  // assert(batch == 1);
    int32_t in_ch = input->dim[1];
    int32_t out_ch = kernel->dim[0];
    int32_t out_h = output->dim[2];
    int32_t out_w = output->dim[3];

    int32_t m = out_ch / group;
    int32_t k = in_ch / group;
    int32_t n = out_h * out_w;

    float *pb_reorder = (float *)shl_mem_alloc(k * n * sizeof(float));

    // float *input_ncxhwx = (float *)shl_mem_alloc(k * n * sizeof(float));
    // float *output_ncxhwx = (float *)shl_mem_alloc(m * n * sizeof(float));

    for (int i = 0; i < batch; i++) {
        for (int g = 0; g < group; g++) {
            float *kernel_ptr = kernel_data + g * m * k;
            float *in_ptr = pb_reorder;
            float *out_ptr = output_data;
            float *bias_ptr = bias_data ? (bias_data + g * m) : NULL;

            // shl_rvv_reorder_input_pack1ton_fp32(input_data, input_ncxhwx, k, out_h, out_w);

            // pack
            shl_rvv_reorder_input_z12_packn_fp32(input_data, in_ptr, k, n, n);
            // GEMM
            // shl_rvv_ncxhwx_gemm_12xpack2n_fp32(pc, pa, pb, m, k, n, n, bias_data + g * m);
            shl_c908_ncxhwx_gemm_12xpack2n_fp32(out_ptr, kernel_ptr, in_ptr, bias_ptr, m, k, n,
                                                false);

            // shl_rvv_reorder_input_packnto1_fp32(output_ncxhwx, output_data, m, out_h, out_w);

            input_data += k * n;
            output_data += m * n;
        }
    }
    // shl_mem_free(input_ncxhwx);
    // shl_mem_free(output_ncxhwx);
    shl_mem_free(pb_reorder);
    return CSINN_TRUE;
}
