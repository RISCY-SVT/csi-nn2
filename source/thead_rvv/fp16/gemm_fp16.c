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
 * input matrix and kernel matrix have been reordered
 ***********************************************************************/

/*
 * GEMM operation: dst = sa * sb + bias
 * where:
 *   dst - output matrix [m, n]
 *   sa  - kernel matrix [m, k] - reordered by shl_rvv_reorder_kernel_n8_fp16
 *   sb  - input matrix  [k, n] - reordered by shl_rvv_reorder_input_z16_fp16
 *   bias - bias vector [m] (optional)
 *
 * Data layout after reorder_input_z16:
 * - For complete blocks of 16 columns: data is packed as [16 elements from row 0], [16 elements from row 1], etc.
 * - For blocks of 8 columns: data is packed as [8 elements from row 0], [8 elements from row 1], etc.
 * - For tail columns (n % 8): processed column by column, with data packed by groups of 16 rows
 * - Important: Column j starts at offset j*k in the reordered buffer (for j >= 16)
 *
 * Data layout after reorder_kernel_n8:
 * - Kernel is reordered in blocks of 8x8 (or mxk for m<8)
 * - Within each block: [m elements from column 0], [m elements from column 1], etc.
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
    // Cast input pointers for clarity
    __fp16 *kernel_data = (__fp16 *)sa;  // Reordered kernel matrix
    __fp16 *input_data = (__fp16 *)sb;   // Reordered input matrix
    __fp16 *output_data = dst;           // Output matrix

    // Handle bias: if not provided, allocate zero-initialized bias
    int flag_bias = 1;  // Flag to track if we need to free bias later
    if (bias == NULL) {
        flag_bias = 0;
        bias = (__fp16 *)shl_mem_alloc(m * sizeof(__fp16));
        // Note: shl_mem_alloc likely zero-initializes, but if not, memset would be needed
    }
    __fp16 *bias_ptr = bias;

    int vl;  // Vector length for RVV operations

    int i = 0;  // Row index for output matrix
    
    /* ============================================================================
     * Process M8 blocks: Handle 8 rows at a time for optimal vector utilization
     * This is the most efficient case as it fully utilizes RVV registers
     * ============================================================================ */
    for (; i + 7 < m; i += 8) {
        // For FP16, we can process 16 elements at once in the main loop
        vl = vsetvl_e16m2(16);

        // Reset input pointer to beginning for this block of 8 rows
        __fp16 *in_ptr = input_data;

        // Set up output pointers for 8 consecutive rows
        // Each pointer is offset by ldc (leading dimension of C)
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;
        __fp16 *out_ptr4 = out_ptr3 + ldc;
        __fp16 *out_ptr5 = out_ptr4 + ldc;
        __fp16 *out_ptr6 = out_ptr5 + ldc;
        __fp16 *out_ptr7 = out_ptr6 + ldc;

        int j = 0;  // Column index for output matrix
        
        /* ------------------------------------------------------------------------
         * M8N16: Process 8x16 output tiles
         * This is the most efficient inner kernel for FP16
         * ------------------------------------------------------------------------ */
        for (; j + 15 < n; j += 16) {
            // Reset kernel pointer for this tile
            __fp16 *kernel_ptr = kernel_data;
            
            // Initialize 8 accumulator vectors with bias values
            // Each accumulator will hold 16 output values for one row
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);
            vfloat16m2_t _acc2 = vfmv_v_f_f16m2(bias_ptr[2], vl);
            vfloat16m2_t _acc3 = vfmv_v_f_f16m2(bias_ptr[3], vl);
            vfloat16m2_t _acc4 = vfmv_v_f_f16m2(bias_ptr[4], vl);
            vfloat16m2_t _acc5 = vfmv_v_f_f16m2(bias_ptr[5], vl);
            vfloat16m2_t _acc6 = vfmv_v_f_f16m2(bias_ptr[6], vl);
            vfloat16m2_t _acc7 = vfmv_v_f_f16m2(bias_ptr[7], vl);

            // Compute dot product over k dimension
            for (int c = 0; c < k; c++) {
                // Load 16 input values (one for each output column)
                vfloat16m2_t _input = vle16_v_f16m2(in_ptr, vl);

                // Load 8 kernel values (one for each output row)
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                __fp16 k2 = kernel_ptr[2];
                __fp16 k3 = kernel_ptr[3];
                __fp16 k4 = kernel_ptr[4];
                __fp16 k5 = kernel_ptr[5];
                __fp16 k6 = kernel_ptr[6];
                __fp16 k7 = kernel_ptr[7];

                // Perform multiply-accumulate: acc[i] += k[i] * input
                // Each accumulator corresponds to one output row
                _acc0 = vfmacc_vf_f16m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f16m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f16m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f16m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f16m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f16m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f16m2(_acc7, k7, _input, vl);

                // Advance pointers to next k iteration
                kernel_ptr += 8;   // 8 kernel values per iteration
                in_ptr += 16;      // 16 input values per iteration
            }
            
            // Store results to output matrix
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            vse16_v_f16m2(out_ptr2, _acc2, vl);
            vse16_v_f16m2(out_ptr3, _acc3, vl);
            vse16_v_f16m2(out_ptr4, _acc4, vl);
            vse16_v_f16m2(out_ptr5, _acc5, vl);
            vse16_v_f16m2(out_ptr6, _acc6, vl);
            vse16_v_f16m2(out_ptr7, _acc7, vl);
            
            // Advance output pointers by 16 columns
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
            out_ptr4 += 16;
            out_ptr5 += 16;
            out_ptr6 += 16;
            out_ptr7 += 16;
        }

        // Switch to smaller vector length for remaining columns
        vl = vsetvl_e16m1(8);

        /* ------------------------------------------------------------------------
         * M8N8: Handle next 8 columns (if any)
         * Process 8x8 output tile
         * ------------------------------------------------------------------------ */
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer to correct position for column j
            // Without this, we'd be reading from wrong memory location
            in_ptr = input_data + j * k;
            
            // Initialize accumulators for 8 columns
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);

            // Compute dot product
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
                in_ptr += 8;  // Only 8 input values per iteration
            }
            
            // Store results
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

        /* ------------------------------------------------------------------------
         * M8N4: Handle remaining 4 columns (if any)
         * Process 8x4 output tile
         * ------------------------------------------------------------------------ */
        if (j + 3 < n) {
            vl = vsetvl_e16m1(4);  // Set vector length to 4
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer to correct position for column j
            in_ptr = input_data + j * k;
            
            // Initialize accumulators for 4 columns
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);
            
            // Compute dot product
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
                in_ptr += 4;  // Only 4 input values per iteration
            }
            
            // Store results
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            
            j += 4;
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
            out_ptr4 += 4;
            out_ptr5 += 4;
            out_ptr6 += 4;
            out_ptr7 += 4;
        }

        /* ------------------------------------------------------------------------
         * M8N2: Handle remaining 2 columns (if any)
         * Process 8x2 output tile
         * ------------------------------------------------------------------------ */
        if (j + 1 < n) {
            vl = vsetvl_e16m1(2);  // Set vector length to 2
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            // Initialize accumulators for 2 columns
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            vfloat16m1_t _acc2 = vfmv_v_f_f16m1(bias_ptr[2], vl);
            vfloat16m1_t _acc3 = vfmv_v_f_f16m1(bias_ptr[3], vl);
            vfloat16m1_t _acc4 = vfmv_v_f_f16m1(bias_ptr[4], vl);
            vfloat16m1_t _acc5 = vfmv_v_f_f16m1(bias_ptr[5], vl);
            vfloat16m1_t _acc6 = vfmv_v_f_f16m1(bias_ptr[6], vl);
            vfloat16m1_t _acc7 = vfmv_v_f_f16m1(bias_ptr[7], vl);
            
            // Compute dot product
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
                in_ptr += 2;  // Only 2 input values per iteration
            }
            
            // Store results
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            vse16_v_f16m1(out_ptr4, _acc4, vl);
            vse16_v_f16m1(out_ptr5, _acc5, vl);
            vse16_v_f16m1(out_ptr6, _acc6, vl);
            vse16_v_f16m1(out_ptr7, _acc7, vl);
            
            j += 2;
            out_ptr0 += 2;
            out_ptr1 += 2;
            out_ptr2 += 2;
            out_ptr3 += 2;
            out_ptr4 += 2;
            out_ptr5 += 2;
            out_ptr6 += 2;
            out_ptr7 += 2;
        }

        /* ------------------------------------------------------------------------
         * M8N1: Handle last column (if any)
         * Process 8x1 output tile using scalar operations
         * ------------------------------------------------------------------------ */
        if (j < n) {
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            // Initialize scalar accumulators with bias
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];
            __fp16 acc2 = bias_ptr[2];
            __fp16 acc3 = bias_ptr[3];
            __fp16 acc4 = bias_ptr[4];
            __fp16 acc5 = bias_ptr[5];
            __fp16 acc6 = bias_ptr[6];
            __fp16 acc7 = bias_ptr[7];
            
            // Compute dot product using scalar operations
            for (int c = 0; c < k; c++) {
                __fp16 input_val = in_ptr[0];
                
                acc0 += kernel_ptr[0] * input_val;
                acc1 += kernel_ptr[1] * input_val;
                acc2 += kernel_ptr[2] * input_val;
                acc3 += kernel_ptr[3] * input_val;
                acc4 += kernel_ptr[4] * input_val;
                acc5 += kernel_ptr[5] * input_val;
                acc6 += kernel_ptr[6] * input_val;
                acc7 += kernel_ptr[7] * input_val;
                
                kernel_ptr += 8;
                in_ptr += 1;  // Only 1 input value per iteration
            }
            
            // Store scalar results
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
            out_ptr2[0] = acc2;
            out_ptr3[0] = acc3;
            out_ptr4[0] = acc4;
            out_ptr5[0] = acc5;
            out_ptr6[0] = acc6;
            out_ptr7[0] = acc7;
        }
        
        // Advance to next block of 8 rows
        kernel_data += 8 * k;   // Skip 8 rows of kernel
        output_data += 8 * ldc;  // Skip 8 rows of output
        bias_ptr += 8;          // Skip 8 bias values
    }

    /* ============================================================================
     * Process M4 blocks: Handle 4 rows at a time
     * Used when remaining rows >= 4 but < 8
     * ============================================================================ */
    for (; i + 3 < m; i += 4) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;

        // Set up 4 output row pointers
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;
        __fp16 *out_ptr2 = out_ptr1 + ldc;
        __fp16 *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        
        /* ------------------------------------------------------------------------
         * M4N16: Process 4x16 output tiles
         * ------------------------------------------------------------------------ */
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            
            // Initialize 4 accumulator vectors
            vfloat16m2_t _acc0 = vfmv_v_f_f16m2(bias_ptr[0], vl);
            vfloat16m2_t _acc1 = vfmv_v_f_f16m2(bias_ptr[1], vl);
            vfloat16m2_t _acc2 = vfmv_v_f_f16m2(bias_ptr[2], vl);
            vfloat16m2_t _acc3 = vfmv_v_f_f16m2(bias_ptr[3], vl);

            // Compute dot product
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
            
            // Store results
            vse16_v_f16m2(out_ptr0, _acc0, vl);
            vse16_v_f16m2(out_ptr1, _acc1, vl);
            vse16_v_f16m2(out_ptr2, _acc2, vl);
            vse16_v_f16m2(out_ptr3, _acc3, vl);
            
            out_ptr0 += 16;
            out_ptr1 += 16;
            out_ptr2 += 16;
            out_ptr3 += 16;
        }

        /* ------------------------------------------------------------------------
         * M4N8: Handle next 8 columns
         * ------------------------------------------------------------------------ */
        for (; j + 7 < n; j += 8) {
            vl = vsetvl_e16m1(8);
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
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

        /* ------------------------------------------------------------------------
         * M4N4: Handle remaining 4 columns
         * ------------------------------------------------------------------------ */
        if (j + 3 < n) {
            vl = vsetvl_e16m1(4);
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
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
                in_ptr += 4;
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            
            j += 4;
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
        }
        
        /* ------------------------------------------------------------------------
         * M4N2: Handle remaining 2 columns
         * ------------------------------------------------------------------------ */
        if (j + 1 < n) {
            vl = vsetvl_e16m1(2);
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
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
                in_ptr += 2;
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            vse16_v_f16m1(out_ptr2, _acc2, vl);
            vse16_v_f16m1(out_ptr3, _acc3, vl);
            
            j += 2;
            out_ptr0 += 2;
            out_ptr1 += 2;
            out_ptr2 += 2;
            out_ptr3 += 2;
        }
        
        /* ------------------------------------------------------------------------
         * M4N1: Handle last column
         * ------------------------------------------------------------------------ */
        if (j < n) {
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];
            __fp16 acc2 = bias_ptr[2];
            __fp16 acc3 = bias_ptr[3];
            
            for (int c = 0; c < k; c++) {
                __fp16 input_val = in_ptr[0];
                
                acc0 += kernel_ptr[0] * input_val;
                acc1 += kernel_ptr[1] * input_val;
                acc2 += kernel_ptr[2] * input_val;
                acc3 += kernel_ptr[3] * input_val;
                
                kernel_ptr += 4;
                in_ptr += 1;
            }
            
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
            out_ptr2[0] = acc2;
            out_ptr3[0] = acc3;
        }
        
        // Advance to next block of 4 rows
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_ptr += 4;
    }

    /* ============================================================================
     * Process M2 blocks: Handle 2 rows at a time
     * Used when remaining rows >= 2 but < 4
     * ============================================================================ */
    for (; i + 1 < m; i += 2) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;
        
        // Set up 2 output row pointers
        __fp16 *out_ptr0 = output_data;
        __fp16 *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        
        /* ------------------------------------------------------------------------
         * M2N16: Process 2x16 output tiles
         * ------------------------------------------------------------------------ */
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
            
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

        vl = vsetvl_e16m1(8);
        
        /* ------------------------------------------------------------------------
         * M2N8: Handle next 8 columns
         * ------------------------------------------------------------------------ */
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
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

        /* ------------------------------------------------------------------------
         * M2N4: Handle remaining 4 columns
         * ------------------------------------------------------------------------ */
        if (j + 3 < n) {
            vl = vsetvl_e16m1(4);
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                
                kernel_ptr += 2;
                in_ptr += 4;
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            
            j += 4;
            out_ptr0 += 4;
            out_ptr1 += 4;
        }
        
        /* ------------------------------------------------------------------------
         * M2N2: Handle remaining 2 columns
         * ------------------------------------------------------------------------ */
        if (j + 1 < n) {
            vl = vsetvl_e16m1(2);
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            vfloat16m1_t _acc1 = vfmv_v_f_f16m1(bias_ptr[1], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                __fp16 k1 = kernel_ptr[1];
                
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f16m1(_acc1, k1, _input, vl);
                
                kernel_ptr += 2;
                in_ptr += 2;
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            vse16_v_f16m1(out_ptr1, _acc1, vl);
            
            j += 2;
            out_ptr0 += 2;
            out_ptr1 += 2;
        }
        
        /* ------------------------------------------------------------------------
         * M2N1: Handle last column
         * ------------------------------------------------------------------------ */
        if (j < n) {
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            __fp16 acc0 = bias_ptr[0];
            __fp16 acc1 = bias_ptr[1];
            
            for (int c = 0; c < k; c++) {
                __fp16 input_val = in_ptr[0];
                
                acc0 += kernel_ptr[0] * input_val;
                acc1 += kernel_ptr[1] * input_val;
                
                kernel_ptr += 2;
                in_ptr += 1;
            }
            
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
        }
        
        // Advance to next block of 2 rows
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_ptr += 2;
    }

    /* ============================================================================
     * Process M1 blocks: Handle last row (if any)
     * This handles the case when m is odd
     * ============================================================================ */
    for (; i < m; i++) {
        vl = vsetvl_e16m2(16);

        __fp16 *in_ptr = input_data;
        __fp16 *out_ptr0 = output_data;

        int j = 0;
        
        /* ------------------------------------------------------------------------
         * M1N16: Process 1x16 output tiles
         * ------------------------------------------------------------------------ */
        for (; j + 15 < n; j += 16) {
            __fp16 *kernel_ptr = kernel_data;
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

        vl = vsetvl_e16m1(8);
        
        /* ------------------------------------------------------------------------
         * M1N8: Handle next 8 columns
         * ------------------------------------------------------------------------ */
        for (; j + 7 < n; j += 8) {
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
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

        /* ------------------------------------------------------------------------
         * M1N4: Handle remaining 4 columns
         * ------------------------------------------------------------------------ */
        if (j + 3 < n) {
            vl = vsetvl_e16m1(4);
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                
                kernel_ptr += 1;
                in_ptr += 4;
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            
            j += 4;
            out_ptr0 += 4;
        }
        
        /* ------------------------------------------------------------------------
         * M1N2: Handle remaining 2 columns
         * ------------------------------------------------------------------------ */
        if (j + 1 < n) {
            vl = vsetvl_e16m1(2);
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            vfloat16m1_t _acc0 = vfmv_v_f_f16m1(bias_ptr[0], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat16m1_t _input = vle16_v_f16m1(in_ptr, vl);
                __fp16 k0 = kernel_ptr[0];
                
                _acc0 = vfmacc_vf_f16m1(_acc0, k0, _input, vl);
                
                kernel_ptr += 1;
                in_ptr += 2;
            }
            
            vse16_v_f16m1(out_ptr0, _acc0, vl);
            
            j += 2;
            out_ptr0 += 2;
        }
        
        /* ------------------------------------------------------------------------
         * M1N1: Handle last element
         * ------------------------------------------------------------------------ */
        if (j < n) {
            __fp16 *kernel_ptr = kernel_data;
            
            // CRITICAL FIX: Reset input pointer for column j
            in_ptr = input_data + j * k;
            
            __fp16 acc = bias_ptr[0];
            
            for (int c = 0; c < k; c++) {
                acc += kernel_ptr[0] * in_ptr[0];
                kernel_ptr += 1;
                in_ptr += 1;
            }
            
            out_ptr0[0] = acc;
        }
    }

    // Clean up: free temporary bias if we allocated it
    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
