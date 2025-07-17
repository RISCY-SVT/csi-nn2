/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
 * Copyright (C) 2025 Sergey V. Tyurin
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

#include "rvv/rvv.h"

/************************************************************************
 * input matrix and kernel matrix have been reordered
 ***********************************************************************/

/*
 * Return byte offset of column @col in reordered buffer (layout Z16/Z8/Z4/Z2/Z1).
 * Rule:   – сначала прибавляем k*BLOCK  за все полные блоки левее,
 *         – внутри текущего блока прибавляем просто (col_in_block).
 */
static inline size_t sb_offset_z16(size_t k, int col)
{
    size_t off = 0;

    /* 1. Полные блоки 16-8-4-2 левее текущей колонки */
    while (col >= 16) { off += k * 16; col -= 16; }
    if (col >= 8) { off += k * 8;  col -= 8;  }
    if (col >= 4) { off += k * 4;  col -= 4;  }
    if (col >= 2) { off += k * 2;  col -= 2;  }

    /* 2. Смещение внутри блока */
    return off + col;          /* col ∈ {0,1} */
}

/*
 * GEMM operation: dst = sa * sb + bias
 * where:
 *   dst - output matrix [m, n]
 *   sa  - kernel matrix [m, k] - reordered by shl_rvv_reorder_kernel_n8_fp16
 *   sb  - input matrix  [k, n] - reordered by shl_rvv_reorder_input_z16_fp16
 *   bias - bias vector [m] (optional)
 *
 * Data layout after reorder_input_z16:
 * - Data is packed in row-major format within variable-sized blocks
 * - Block sizes follow pattern: 16, 8, 4, 2, 1 (as needed for tail)
 * - Each block contains all k rows for its columns stored contiguously
 *
 * Processing strategy:
 * - Process output in tiles: m8n16, m8n8, m8n4, m8n2, m8n1, then m4n*, m2n*, m1n*
 * - Use RVV vector instructions for efficient computation
 * - Handle edge cases (tail rows/columns) with smaller tiles
 * - FP16 version processes larger N tiles (16) compared to FP32 (8) due to double vector capacity
 */
void shl_rvv_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias, int m,
                            int k, int n, int ldc)
{
    printf("[GEMM_FP16 DEBUG] Entry: dst=%p, sa=%p, sb=%p, bias=%p\n", dst, sa, sb, bias);
    printf("[GEMM_FP16 DEBUG] Params: m=%d, k=%d, n=%d, ldc=%d\n", m, k, n, ldc);
    
    // check for critical cases
    if (n == 3) printf("[GEMM_FP16 DEBUG] Critical case: n=3\n");
    if (n == 5) printf("[GEMM_FP16 DEBUG] Critical case: n=5\n");
    if (n == 6) printf("[GEMM_FP16 DEBUG] Critical case: n=6\n");
    if (n == 7) printf("[GEMM_FP16 DEBUG] Critical case: n=7\n");
    
    // Add debug output for first few calls
    static int call_count = 0;
    if (++call_count <= 10) {
        printf("[GEMM_FP16] Call %d: m=%d, k=%d, n=%d, ldc=%d\n", call_count, m, k, n, ldc);
    }
    
    __fp16 *kernel_data = (__fp16 *)sa;
    __fp16 *input_data = (__fp16 *)sb;
    __fp16 *output_data = dst;

    int flag_bias = 1;
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * sizeof(__fp16));
    }
    __fp16 *bias_ptr = bias;

    int vl;

    int i = 0;
    // m8 blocks
    for (; i + 7 < m; i += 8) {
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = output_data + ldc;
        __fp16 *out_ptr2 = output_data + 2 * ldc;
        __fp16 *out_ptr3 = output_data + 3 * ldc;
        __fp16 *out_ptr4 = output_data + 4 * ldc;
        __fp16 *out_ptr5 = output_data + 5 * ldc;
        __fp16 *out_ptr6 = output_data + 6 * ldc;
        __fp16 *out_ptr7 = output_data + 7 * ldc;

        int j = 0;
        
        // m8n16 loop - process full 16-column blocks
        vl = vsetvl_e16m2(16);
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);
            vfloat16m2_t _acc2 = vfmv_v_f_f16m2(bias_ptr[2], vl);
            vfloat16m2_t _acc3 = vfmv_v_f_f16m2(bias_ptr[3], vl);
            vfloat16m2_t _acc4 = vfmv_v_f_f16m2(bias_ptr[4], vl);
            vfloat16m2_t _acc5 = vfmv_v_f_f16m2(bias_ptr[5], vl);
            vfloat16m2_t _acc6 = vfmv_v_f_f16m2(bias_ptr[6], vl);
            vfloat16m2_t _acc7 = vfmv_v_f_f16m2(bias_ptr[7], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);
                in_ptr += 16;  // Move to next row in 16-block
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];
                kernel_ptr += 8;

                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m2(_acc7, k7, _input, vl);
            }
            
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            vse16_v_f16m2(out_ptr2, _acc2, vl);
            vse16_v_f16m2(out_ptr3, _acc3, vl);
            vse16_v_f16m2(out_ptr4, _acc4, vl);
            vse16_v_f16m2(out_ptr5, _acc5, vl);
            vse16_v_f16m2(out_ptr6, _acc6, vl);
            vse16_v_f16m2(out_ptr7, _acc7, vl);
            
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
            out_ptr4 += 16;
            out_ptr5 += 16;
            out_ptr6 += 16;
            out_ptr7 += 16;
        }

        // m8n8 loop - process 8-column blocks
        vl = vsetvl_e16m1(8);
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 8;  // Move to next row in 8-block
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];
                kernel_ptr += 8;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, k7, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
            out_ptr4 += 8;
            out_ptr5 += 8;
            out_ptr6 += 8;
            out_ptr7 += 8;
        }

        // m8n4 loop - process 4-column blocks
        vl = vsetvl_e16m1(4);
        for (; j + 3 < n; j += 4) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 4;  // Move to next row in 4-block
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];
                kernel_ptr += 8;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, k7, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
            out_ptr4 += 4;
            out_ptr5 += 4;
            out_ptr6 += 4;
            out_ptr7 += 4;
        }

        // m8n2 loop - process 2-column blocks
        vl = vsetvl_e16m1(2);
        for (; j + 1 < n; j += 2) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 2;  // Move to next row in 2-block
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];
                kernel_ptr += 8;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, k7, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            
            out_ptr0 += 2;
            out_ptr1 += 2;
            out_ptr2 += 2;
            out_ptr3 += 2;
            out_ptr4 += 2;
            out_ptr5 += 2;
            out_ptr6 += 2;
            out_ptr7 += 2;
        }

        // m8n1 loop - process single columns
        vl = vsetvl_e16m1(1);
        for (; j < n; j++) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];
            __fp16 acc2 = bias_ptr[2];
            __fp16 acc3 = bias_ptr[3];
            __fp16 acc4 = bias_ptr[4];
            __fp16 acc5 = bias_ptr[5];
            __fp16 acc6 = bias_ptr[6];
            __fp16 acc7 = bias_ptr[7];

            for (int c = 0; c < k; c++) {
                __fp16 input_val = in_ptr[0];
                in_ptr += 1;  // Move to next row in 1-block
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];
                kernel_ptr += 8;

                acc0 += k0 * input_val;
                acc1 += k1 * input_val;
                acc2 += k2 * input_val;
                acc3 += k3 * input_val;
                acc4 += k4 * input_val;
                acc5 += k5 * input_val;
                acc6 += k6 * input_val;
                acc7 += k7 * input_val;
            }
            
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
            out_ptr2[0] = acc2;
            out_ptr3[0] = acc3;
            out_ptr4[0] = acc4;
            out_ptr5[0] = acc5;
            out_ptr6[0] = acc6;
            out_ptr7[0] = acc7;
            
            out_ptr0++;
            out_ptr1++;
            out_ptr2++;
            out_ptr3++;
            out_ptr4++;
            out_ptr5++;
            out_ptr6++;
            out_ptr7++;
        }

        bias_ptr += 8;
        output_data += 8 * ldc;
        kernel_data += 8 * k;
    }

    // m4 blocks
    for (; i + 3 < m; i += 4) {
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = output_data + ldc;
        __fp16 *out_ptr2 = output_data + 2 * ldc;
        __fp16 *out_ptr3 = output_data + 3 * ldc;

        int j = 0;
        
        // m4n16 loop
        vl = vsetvl_e16m2(16);
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);
            vfloat16m2_t _acc2 = vfmv_v_f_f16m2(bias_ptr[2], vl);
            vfloat16m2_t _acc3 = vfmv_v_f_f16m2(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);
                in_ptr += 16;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                kernel_ptr += 4;

                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, k3, _input, vl);
            }
            
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            vse16_v_f16m2(out_ptr2, _acc2, vl);
            vse16_v_f16m2(out_ptr3, _acc3, vl);
            
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
        }

        // m4n8 loop
        vl = vsetvl_e16m1(8);
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 8;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                kernel_ptr += 4;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }

        // m4n4 loop
        vl = vsetvl_e16m1(4);
        for (; j + 3 < n; j += 4) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 4;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                kernel_ptr += 4;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
        }

        // m4n2 loop
        vl = vsetvl_e16m1(2);
        for (; j + 1 < n; j += 2) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 2;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                kernel_ptr += 4;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            
            out_ptr0 += 2;
            out_ptr1 += 2;
            out_ptr2 += 2;
            out_ptr3 += 2;
        }

        // m4n1 loop
        for (; j < n; j++) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];
            __fp16 acc2 = bias_ptr[2];
            __fp16 acc3 = bias_ptr[3];

            for (int c = 0; c < k; c++) {
                __fp16 input_val = in_ptr[0];
                in_ptr += 1;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                kernel_ptr += 4;

                acc0 += k0 * input_val;
                acc1 += k1 * input_val;
                acc2 += k2 * input_val;
                acc3 += k3 * input_val;
            }
            
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
            out_ptr2[0] = acc2;
            out_ptr3[0] = acc3;
            
            out_ptr0++;
            out_ptr1++;
            out_ptr2++;
            out_ptr3++;
        }

        bias_ptr += 4;
        output_data += 4 * ldc;
        kernel_data += 4 * k;
    }

    // m2 blocks
    for (; i + 1 < m; i += 2) {
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = output_data + ldc;

        int j = 0;
        
        // m2n16 loop
        vl = vsetvl_e16m2(16);
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);
                in_ptr += 16;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                kernel_ptr += 2;

                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
            }
            
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            
            out_ptr0 += 16;
            out_ptr1 += 16;
        }

        // m2n8 loop
        vl = vsetvl_e16m1(8);
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 8;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                kernel_ptr += 2;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            
            out_ptr0 += 8;
            out_ptr1 += 8;
        }

        // m2n4 loop
        vl = vsetvl_e16m1(4);
        for (; j + 3 < n; j += 4) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 4;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                kernel_ptr += 2;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            
            out_ptr0 += 4;
            out_ptr1 += 4;
        }

        // m2n2 loop
        vl = vsetvl_e16m1(2);
        for (; j + 1 < n; j += 2) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 2;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                kernel_ptr += 2;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            
            out_ptr0 += 2;
            out_ptr1 += 2;
        }

        // m2n1 loop
        for (; j < n; j++) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];

            for (int c = 0; c < k; c++) {
                __fp16 input_val = in_ptr[0];
                in_ptr += 1;
                
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                kernel_ptr += 2;

                acc0 += k0 * input_val;
                acc1 += k1 * input_val;
            }
            
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
            
            out_ptr0++;
            out_ptr1++;
        }

        bias_ptr += 2;
        output_data += 2 * ldc;
        kernel_data += 2 * k;
    }

    // m1 blocks
    for (; i < m; i++) {
        __fp16 *out_ptr0 = output_data;

        int j = 0;
        
        // m1n16 loop
        vl = vsetvl_e16m2(16);
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);
                in_ptr += 16;
                
                __fp16 k0 = kernel_ptr[0];
                kernel_ptr++;

                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
            }
            
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            out_ptr0 += 16;
        }

        // m1n8 loop
        vl = vsetvl_e16m1(8);
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 8;
                
                __fp16 k0 = kernel_ptr[0];
                kernel_ptr++;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            out_ptr0 += 8;
        }

        // m1n4 loop
        vl = vsetvl_e16m1(4);
        for (; j + 3 < n; j += 4) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 4;
                
                __fp16 k0 = kernel_ptr[0];
                kernel_ptr++;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            out_ptr0 += 4;
        }

        // m1n2 loop
        vl = vsetvl_e16m1(2);
        for (; j + 1 < n; j += 2) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                in_ptr += 2;
                
                __fp16 k0 = kernel_ptr[0];
                kernel_ptr++;

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            out_ptr0 += 2;
        }

        // m1n1 loop
        for (; j < n; j++) {
            __fp16 *kernel_ptr = kernel_data;
            __fp16 *in_ptr = input_data + sb_offset_z16(k, j);
            
            __fp16 acc0 = bias_ptr[0];

            for (int c = 0; c < k; c++) {
                __fp16 input_val = in_ptr[0];
                in_ptr += 1;
                
                __fp16 k0 = kernel_ptr[0];
                kernel_ptr++;

                acc0 += k0 * input_val;
            }
            
            out_ptr0[0] = acc0;
            out_ptr0++;
        }

        bias_ptr++;
        output_data += ldc;
        kernel_data += k;
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
