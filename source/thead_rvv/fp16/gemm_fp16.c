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

#include "rvv/rvv.h"

/************************************************************************
 * GEMM (General Matrix Multiply) for FP16 with RVV 0.7.1
 * 
 * This implementation uses a hybrid data layout approach:
 * 1. Full blocks (16/8 columns): Row-major format
 *    - Data arranged as: [row0][row1]...[rowK-1]
 *    - Each row contains N consecutive elements
 *    - Enables efficient vector loads/stores
 * 
 * 2. Tail blocks (<8 columns): Column-major format
 *    - Data arranged as: [col0][col1]...[colN-1]
 *    - Each column contains K consecutive elements
 *    - Maintains compatibility with other library functions
 * 
 * The hybrid approach balances performance (vectorization for main blocks)
 * with flexibility (column-major for tail handling).
 * 
 * Matrix dimensions:
 * - dst: Output matrix [m × n]
 * - sa:  Kernel matrix [m × k] (reordered by shl_rvv_reorder_kernel_n8_fp16)
 * - sb:  Input matrix  [k × n] (reordered by shl_rvv_reorder_input_z16_fp16)
 * - bias: Optional bias vector [m]
 * 
 * Processing tiles:
 * - Main tiles: m8n16, m8n8, m4n16, m4n8, m2n16, m2n8, m1n16, m1n8
 * - Tail tiles: mXn4, mXn2, mXn1 (where X ∈ {8,4,2,1})
 * 
 * Performance notes:
 * - Main blocks use vector loads/stores for maximum throughput
 * - Tail blocks use scalar loads with vector accumulation for correctness
 * - m2 and m1 tail blocks use optimized vector operations where possible
 * - Strided stores (vsse) are used for tail blocks to maintain output layout
 * 
 * Design rationale for column-major tails:
 * - Simplifies indexing: column_ptr = base + col * k
 * - Maintains compatibility with legacy code and other functions
 * - Allows uniform handling across different tail sizes
 * 
 * CRITICAL: 8-column blocks remain row-major even as tails after 16-blocks!
 ***********************************************************************/

// vlen=128
void shl_rvv_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, __fp16 *bias, int m,
                            int k, int n, int ldc)
{
    __fp16 *kernel_data = (__fp16 *)sa;
    __fp16 *input_data = (__fp16 *)sb;
    __fp16 *output_data = dst;

    int flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * sizeof(__fp16));
    }
    __fp16 *bias_ptr = bias;

    int vl;

    int i = 0;
    // m8 loop - process 8 output rows at a time
    for (; i + 7 < m; i += 8) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;

        // Set up output row pointers
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;
        __fp16 *out_ptr4 = out_ptr3 + ldc;
        __fp16 *out_ptr5 = out_ptr4 + ldc;
        __fp16 *out_ptr6 = out_ptr5 + ldc;
        __fp16 *out_ptr7 = out_ptr6 + ldc;

        int j = 0;
        // m8n16 loop - process 16 columns at a time
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            
            // Initialize accumulators with bias values
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

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];

                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m2(_acc7, k7, _input, vl);

                kernel_ptr += 8;
                in_ptr += 16;
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

        vl = vsetvl_e16m1(8);
        // m8n8 loop - CRITICAL: 8-column blocks are ALWAYS row-major!
        for (; j + 7 < n; j += 8) {
            // Calculate correct pointer for 8-column blocks in hybrid format
            int full_16_blocks = j / 16;
            int blocks_8_in_tail = (j % 16) / 8;
            __fp16 *in_ptr = input_data + full_16_blocks * 16 * k + blocks_8_in_tail * 8 * k;
            
            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            
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

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m1(_acc7, k7, _input, vl);

                kernel_ptr += 8;
                in_ptr += 8;  // Row-major: advance by 8 elements
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

        // m8n4 - column-major format for tail
        for (; j + 3 < n; j += 4) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            
            // Initialize accumulators with bias (broadcast to 8 elements)
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc2 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc3 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer

            // Set up column pointers for column-major access
            __fp16 *in_ptr0 = in_base + 0 * k;
            __fp16 *in_ptr1 = in_base + 1 * k;
            __fp16 *in_ptr2 = in_base + 2 * k;
            __fp16 *in_ptr3 = in_base + 3 * k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            vsse16_v_f16m1(out_ptr2, ldc * sizeof(__fp16), _acc2, vl);
            vsse16_v_f16m1(out_ptr3, ldc * sizeof(__fp16), _acc3, vl);
            out_ptr0 += 4;
        }

        // m8n2 - column-major format for tail
        for (; j + 1 < n; j += 2) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer

            __fp16 *in_ptr0 = in_base + 0 * k;
            __fp16 *in_ptr1 = in_base + 1 * k;

            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            out_ptr0 += 2;
        }

        // m8n1 - column-major format for tail
        for (; j < n; j++) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            __fp16 *in_ptr0 = in_base + 0 * k;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 8;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            out_ptr0 += 1;
        }
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_ptr += 8;
    }

    // m4 - process 4 output rows at a time
    for (; i + 3 < m; i += 4) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        // m4n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);
            vfloat16m2_t _acc2 = vfmv_v_f_f16m2(bias_ptr[2], vl);
            vfloat16m2_t _acc3 = vfmv_v_f_f16m2(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 16;
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

        // m4n8
        vl = vsetvl_e16m1(8);
        for (; j + 7 < n; j += 8) {
            // Calculate correct pointer for 8-column blocks
            int full_16_blocks = j / 16;
            int blocks_8_in_tail = (j % 16) / 8;
            __fp16 *in_ptr = input_data + full_16_blocks * 16 * k + blocks_8_in_tail * 8 * k;

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);

                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];

                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 8;
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

        vl = vsetvl_e16m1(4);
        // m4n4 - column-major for tail
        for (; j + 3 < n; j += 4) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc2 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc3 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer

            __fp16 *in_ptr0 = in_base + 0 * k;
            __fp16 *in_ptr1 = in_base + 1 * k;
            __fp16 *in_ptr2 = in_base + 2 * k;
            __fp16 *in_ptr3 = in_base + 3 * k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f16m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f16m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 4;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            vsse16_v_f16m1(out_ptr2, ldc * sizeof(__fp16), _acc2, vl);
            vsse16_v_f16m1(out_ptr3, ldc * sizeof(__fp16), _acc3, vl);
            out_ptr0 += 4;
        }
        
        // m4n2 - column-major for tail
        for (; j + 1 < n; j += 2) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);
            vfloat16m1_t _acc1 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            __fp16 *in_ptr0 = in_base + 0 * k;
            __fp16 *in_ptr1 = in_base + 1 * k;
            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 4;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            vsse16_v_f16m1(out_ptr1, ldc * sizeof(__fp16), _acc1, vl);
            out_ptr0 += 2;
        }
        
        // m4n1 - column-major for tail
        for (; j < n; j++) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            
            vfloat16m1_t _acc0 = vle16_v_f16m1(bias_ptr, vl);

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            __fp16 *in_ptr0 = in_base + 0 * k;

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f16m1(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 4;
            }
            vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc0, vl);
            out_ptr0 += 1;
        }
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_ptr += 4;
    }

    // m2
    for (; i + 1 < m; i += 2) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;

        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        // m2n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);

                kernel_ptr += 2;
                in_ptr += 16;
            }
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            out_ptr0 += 16;
            out_ptr1 += 16;
        }

        // m2n8 loop
        vl = vsetvl_e16m1(8);
        for (; j + 7 < n; j += 8) {
            // Calculate correct pointer for 8-column blocks
            int full_16_blocks = j / 16;
            int blocks_8_in_tail = (j % 16) / 8;
            __fp16 *in_ptr = input_data + full_16_blocks * 16 * k + blocks_8_in_tail * 8 * k;

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);

                kernel_ptr += 2;
                in_ptr += 8;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            out_ptr0 += 8;
            out_ptr1 += 8;
        }

        // m2 tail - optimized vector operations when possible
        vl = vsetvl_e16m1(2);
        for (; j < n; j++) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            __fp16 *in_ptr0 = in_base + 0 * k;
            
            // Check if ldc is properly aligned for vector operations
            if (ldc % 8 == 0) {
                // Vector path - more efficient
                vfloat16m1_t _acc = vle16_v_f16m1(bias_ptr, vl);
                __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
                
                for (int c = 0; c < k; c++) {
                    vfloat16m1_t _kernel = vle16_v_f16m1(kernel_ptr, vl);
                    _acc = vfmacc_vf_f16m1(_acc, in_ptr0[c], _kernel, vl);
                    kernel_ptr += 2;
                }
                vsse16_v_f16m1(out_ptr0, ldc * sizeof(__fp16), _acc, vl);
            } else {
                // Scalar fallback path
                float acc0 = bias_ptr[0];
                float acc1 = bias_ptr[1];
                
                for (int c = 0; c < k; c++) {
                    // kernel in column-major format
                    acc0 += kernel_data[2 * c] * in_ptr0[c];
                    acc1 += kernel_data[2 * c + 1] * in_ptr0[c];
                }
                *out_ptr0 = acc0;
                *(out_ptr0 + ldc) = acc1;
            }
            out_ptr0++;
        }
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_ptr += 2;
    }

    // m1
    for (; i < m; i++) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;
        __fp16 *out_ptr0 = output_data;

        int j = 0;
        // m1n16 loop
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 16;
            }
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            out_ptr0 += 16;
        }

        // m1n8 loop
        vl = vsetvl_e16m1(8);
        for (; j + 7 < n; j += 8) {
            // Calculate correct pointer for 8-column blocks
            int full_16_blocks = j / 16;
            int blocks_8_in_tail = (j % 16) / 8;
            __fp16 *in_ptr = input_data + full_16_blocks * 16 * k + blocks_8_in_tail * 8 * k;

            __fp16 *kernel_ptr = kernel_data;  // Reset kernel pointer
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);

            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 8;
            }
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            out_ptr0 += 8;
        }

        // m1 tail
        for (; j < n; j++) {
            // Calculate base pointer for column-major tail data
            int full_blocks = (j / 16) * 16 + ((j % 16) / 8) * 8;
            int tail_start = j % ((j / 8) * 8);  // Start of tail within current section
            __fp16 *in_base = input_data + full_blocks * k + tail_start * k;
            __fp16 *in_ptr0 = in_base + 0 * k;
            
            float acc0 = bias_ptr[0];
            
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[c] * in_ptr0[c];
            }
            *out_ptr0++ = acc0;
        }
        kernel_data += 1 * k;
        output_data += 1 * ldc;
        bias_ptr += 1;
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}

// TODO: Verify with unit test - especially edge cases like n=8, n=23, n=31
// The critical fix here is proper calculation of input pointers for the hybrid data format
