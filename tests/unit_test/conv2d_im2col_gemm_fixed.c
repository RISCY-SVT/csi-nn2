/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
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

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"

#define MAX_MISMATCH 5
static void dump_mismatch(const float *out, const float *ref, int size, float tol)
{
    int printed = 0;
    for (int i = 0; i < size && printed < MAX_MISMATCH; ++i) {
        float diff = fabsf(out[i] - ref[i]);
        if (diff > tol) {
            printf("  idx %d : out=%f  ref=%f  |diff|=%f\n", i, out[i], ref[i], diff);
            ++printed;
        }
    }
}

static void verify_conv2d_im2col_reorder(void *kernel_data, void *ref_kernel, void (*reorder)(),
                                         int out_ch, int in_ch, int k_h, int k_w,
                                         enum csinn_dtype_enum dtype)
{
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = out_ch;
    kernel->dim[1] = in_ch;
    kernel->dim[2] = k_h;
    kernel->dim[3] = k_w;
    kernel->dim_count = 4;
    kernel->name = "kernel";
    int kernel_size = csinn_tensor_size(kernel);
    kernel->dtype = dtype;
    kernel->layout = CSINN_LAYOUT_OIHW;

    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->base.name = "params";
    params->base.api = CSINN_C920;  // CRITICAL FIX: Set the API!
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;

    kernel->data = kernel_data;

    reorder(kernel, params);
    evaluate_error(kernel->data, ref_kernel, kernel_size, dtype);
    dump_mismatch((const float *)kernel->data, (const float *)ref_kernel, kernel_size, 1e-4f);
    csinn_free_tensor(kernel);
    csinn_free_params(params);
}

static void verify_conv2d_im2col_compute(void *input_data, void *kernel_data, void *bias_data,
                                         void *ref_data, int (*compute)(), int in_c, int in_h, int in_w,
                                         int out_c, int out_h, int out_w, int k_h, int k_w,
                                         enum csinn_dtype_enum dtype)
{
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = in_h;
    input->dim[3] = in_w;
    input->dim_count = 4;
    input->name = "input";
    int in_size = csinn_tensor_size(input);
    input->dtype  = dtype;
    input->layout = CSINN_LAYOUT_NCHW;

    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = out_c;
    kernel->dim[1] = in_c;
    kernel->dim[2] = k_h;
    kernel->dim[3] = k_w;
    kernel->dim_count = 4;
    kernel->name = "kernel";
    kernel->dtype  = dtype;
    kernel->layout = CSINN_LAYOUT_OIHW;

    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = out_c;
    bias->dim_count = 1;
    bias->name = "bias";
    bias->dtype  = dtype;
    bias->layout = CSINN_LAYOUT_O;

    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_c;
    output->dim[2] = out_h;
    output->dim[3] = out_w;
    output->dim_count = 4;
    output->name = "output";
    output->dtype  = dtype;
    output->layout = CSINN_LAYOUT_NCHW;
    int out_size = csinn_tensor_size(output);
    if (in_size <= 0 || out_size <= 0) {
        printf("Invalid input or output tensor size.\n");
        return;
    }

    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->base.name = "params";
    params->base.api = CSINN_C920;  // CRITICAL FIX: Set the API!
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;

    input->data  = input_data;
    kernel->data = kernel_data;
    bias->data   = bias_data;
    output->data = shl_mem_alloc(out_size * sizeof(float));

    compute(input, output, kernel, bias, params);
    evaluate_error(output->data, ref_data, out_size, dtype);
    dump_mismatch((const float *)output->data, (const float *)ref_data, out_size, 1e-3f);
    
    csinn_free_tensor(input);
    shl_mem_free(output->data);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of convolution im2col_gemm for RVV.\n");

    /* FP32 --------------------------------------------------------------- */
    verify_conv2d_im2col_reorder(conv2d_im2col_fp32_ker, conv2d_im2col_fp32_ker1,
                                 shl_rvv_conv_im2col_gemm_reorder_kernel_fp32, 19, 3, 3, 3,
                                 CSINN_DTYPE_FLOAT32);

    verify_conv2d_im2col_compute(conv2d_im2col_fp32_in, conv2d_im2col_fp32_ker1,
                                 conv2d_im2col_fp32_bias, conv2d_im2col_fp32_out,
                                 shl_rvv_conv_im2col_gemm_fp32, 3, 4, 5, 19, 4, 5, 3, 3,
                                 CSINN_DTYPE_FLOAT32);

    /* FP16 --------------------------------------------------------------- */
    verify_conv2d_im2col_reorder(conv2d_im2col_fp16_ker, conv2d_im2col_fp16_ker1,
                                 shl_rvv_conv_im2col_gemm_reorder_kernel_fp16, 19, 3, 3, 3,
                                 CSINN_DTYPE_FLOAT16);

    verify_conv2d_im2col_compute(conv2d_im2col_fp16_in, conv2d_im2col_fp16_ker1,
                                 conv2d_im2col_fp16_bias, conv2d_im2col_fp16_out,
                                 shl_rvv_conv_im2col_gemm_fp16, 3, 4, 5, 19, 4, 5, 3, 3,
                                 CSINN_DTYPE_FLOAT16);

    return done_testing();
}
