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

#include "./valid_data/pad.dat"

#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"

/* --------------------------------------------------------------------------
 * Generic pad verifier: supports FP32 / FP16 / INT8 back-ends.
 * -------------------------------------------------------------------------- */
static void verify_pad(void *input_data,
                       void *ref_data,
                       void *func_void,
                       int  in_c,
                       int  in_h,
                       int  in_w,
                       int  pad_top,
                       int  pad_left,
                       int  pad_down,
                       int  pad_right,
                       enum csinn_dtype_enum dtype)
{
    const int padded_h = in_h + pad_top  + pad_down;
    const int padded_w = in_w + pad_left + pad_right;
    const int out_size = in_c * padded_h * padded_w;

    /* allocate output buffer with matching element size */
    void *out = NULL;
    if (dtype == CSINN_DTYPE_INT8) {
        out = shl_mem_alloc(out_size * sizeof(int8_t));
    } else if (dtype == CSINN_DTYPE_FLOAT16) {
        out = shl_mem_alloc(out_size * sizeof(__fp16));
    } else { /* FP32 */
        out = shl_mem_alloc(out_size * sizeof(float));
    }

    /* dispatch to the correct backend function -------------------------------- */
    if (dtype == CSINN_DTYPE_INT8) {
        typedef void (*pad_int8_t)(const int8_t*, int8_t*, int,int,int,
                                   int,int,int,int,int8_t);
        ((pad_int8_t)func_void)((const int8_t*)input_data, (int8_t*)out,
                                in_c, in_h, in_w,
                                padded_h, padded_w,
                                pad_top, pad_left,
                                (int8_t)0);
    } else if (dtype == CSINN_DTYPE_FLOAT16) {
        typedef void (*pad_fp16_t)(const __fp16*, __fp16*, int,int,int,
                                   int,int,int,int);
        ((pad_fp16_t)func_void)((const __fp16*)input_data, (__fp16*)out,
                                in_c, in_h, in_w,
                                padded_h, padded_w,
                                pad_top, pad_left);
    } else { /* CSINN_DTYPE_FLOAT32 */
        typedef void (*pad_fp32_t)(const float*, float*, int,int,int,
                                   int,int,int,int);
        ((pad_fp32_t)func_void)((const float*)input_data, (float*)out,
                                in_c, in_h, in_w,
                                padded_h, padded_w,
                                pad_top, pad_left);
    }

    /* compare with reference --------------------------------------------------- */
    evaluate_error(out, ref_data, out_size, dtype);

    shl_mem_free(out);
}

/* -------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    init_testsuite("Test function of pad for RVV.\n");

    verify_pad(pad_fp32_in,  pad_fp32_out,
               (void*)shl_rvv_pad_input_fp32,
               3, 4, 19, 1, 1, 1, 1, CSINN_DTYPE_FLOAT32);

    verify_pad(pad_fp16_in,  pad_fp16_out,
               (void*)shl_rvv_pad_input_fp16,
               3, 4, 19, 1, 1, 1, 1, CSINN_DTYPE_FLOAT16);

    verify_pad(pad_int8_in,  pad_int8_out,
               (void*)shl_rvv_pad_input_int8,
               3, 4, 19, 1, 1, 1, 1, CSINN_DTYPE_INT8);

    return done_testing();
}
