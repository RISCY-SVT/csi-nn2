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
 * Return byte offset of column @col in reordered buffer (layout Z8/Z4/Z2/Z1).
 * Rule:   – сначала прибавляем k*BLOCK за все полные блоки левее,
 *         – внутри текущего блока прибавляем просто (col_in_block).
 */
static inline size_t sb_offset_z8(int k, int col)
{
    size_t off = 0;
    
    // Полные блоки левее текущей колонки
    while (col >= 8) { off += k * 8; col -= 8; }
    if (col >= 4) { off += k * 4; col -= 4; }
    if (col >= 2) { off += k * 2; col -= 2; }
    
    // Смещение внутри блока
    return off + col;  // col ∈ {0,1}
}

/*
 * GEMM operation: dst = sa * sb + bias
 * where:
 *   dst - output matrix [m, n]
 *   sa  - kernel matrix [m, k] - reordered by shl_rvv_reorder_kernel_n8_fp32
 *   sb  - input matrix  [k, n] - reordered by shl_rvv_reorder_input_z8_fp32
 *   bias - bias vector [m] (optional)
 *
 * Data layout after reorder_input_z8:
 * - Data is packed in row-major format within variable-sized blocks
 * - Block sizes follow pattern: 8, 4, 2, 1 (as needed for tail)
 * - Each block contains all k rows for its columns stored contiguously
 *
 * Processing strategy:
 * - Process output in tiles: m8n8, m8n4, m8n2, m8n1, then m4n*, m2n*, m1n*
 * - Use RVV vector instructions for efficient computation
 * - Handle edge cases (tail rows/columns) with smaller tiles
 * - FP32 version processes 8-column tiles (vs 16 for FP16) due to register width
 * 
 * CRITICAL FIX: Tail column handling now properly processes composite tails (3,5,6,7)
 * by sequentially applying 4->2->1 sub-blocks instead of single if statements
 */
void shl_rvv_gemm_8x8_fp32(float *dst, const float *sa, const float *sb, float *bias, int m, int k,
                           int n, int ldc)
{
    printf("[GEMM_FP32 DEBUG] Entry: dst=%p, sa=%p, sb=%p, bias=%p\n", dst, sa, sb, bias);
    printf("[GEMM_FP32 DEBUG] Params: m=%d, k=%d, n=%d, ldc=%d\n", m, k, n, ldc);

    // Add debug output similar to FP16 version
    static int call_count = 0;
    if (++call_count <= 10) {
        printf("[GEMM_FP32 DEBUG] Call %d: m=%d, k=%d, n=%d, ldc=%d\n", call_count, m, k, n, ldc);
    }
    
    // Cast input pointers for clarity
    float *kernel_data = (float *)sa;  // Reordered kernel matrix
    float *input_data = (float *)sb;   // Reordered input matrix
    float *output_data = dst;          // Output matrix

    // Handle bias: if not provided, allocate zero-initialized bias
    int flag_bias = 1;  // Flag to track if we need to free bias later
    if (bias == NULL) {
        flag_bias = 0;
        bias = (float *)shl_mem_alloc(m * sizeof(float));
        // Note: shl_mem_alloc likely zero-initializes, but if not, memset would be needed
    }
    float *bias_ptr = bias;

    int vl;  // Vector length for RVV operations

    int i = 0;  // Row index for output matrix
    
    /* ============================================================================
     * Process M8 blocks: Handle 8 rows at a time for optimal vector utilization
     * This is the most efficient case as it fully utilizes RVV registers
     * ============================================================================ */
    for (; i + 7 < m; i += 8) {
        // Set up output pointers for 8 consecutive rows
        // Each pointer is offset by ldc (leading dimension of C)
        float *out_ptr0 = output_data;
        float *out_ptr1 = output_data + ldc;
        float *out_ptr2 = output_data + 2 * ldc;
        float *out_ptr3 = output_data + 3 * ldc;
        float *out_ptr4 = output_data + 4 * ldc;
        float *out_ptr5 = output_data + 5 * ldc;
        float *out_ptr6 = output_data + 6 * ldc;
        float *out_ptr7 = output_data + 7 * ldc;

        int j = 0;  // Column index for output matrix
        
        /* ------------------------------------------------------------------------
         * M8N8: Process 8x8 output tiles
         * This is the most efficient inner kernel
         * ------------------------------------------------------------------------ */
        vl = vsetvl_e32m2(8);
        for (; j + 7 < n; j += 8) {
            // Reset kernel pointer for this tile
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            // Initialize 8 accumulator vectors with bias values
            // Each accumulator will hold 8 output values for one row
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);
            vfloat32m2_t _acc4 = vfmv_v_f_f32m2(bias_ptr[4], vl);
            vfloat32m2_t _acc5 = vfmv_v_f_f32m2(bias_ptr[5], vl);
            vfloat32m2_t _acc6 = vfmv_v_f_f32m2(bias_ptr[6], vl);
            vfloat32m2_t _acc7 = vfmv_v_f_f32m2(bias_ptr[7], vl);

            // Compute dot product over k dimension
            for (int c = 0; c < k; c++) {
                // Load 8 input values (one for each output column)
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                in_ptr += 8;  // Move to next row in 8-block

                // Load 8 kernel values (one for each output row)
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                float k4 = kernel_ptr[4];
                float k5 = kernel_ptr[5];
                float k6 = kernel_ptr[6];
                float k7 = kernel_ptr[7];
                kernel_ptr += 8;

                // Perform multiply-accumulate: acc[i] += k[i] * input
                // Each accumulator corresponds to one output row
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f32m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f32m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f32m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f32m2(_acc7, k7, _input, vl);
            }
            
            // Store results to output matrix
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            vse32_v_f32m2(out_ptr4, _acc4, vl);
            vse32_v_f32m2(out_ptr5, _acc5, vl);
            vse32_v_f32m2(out_ptr6, _acc6, vl);
            vse32_v_f32m2(out_ptr7, _acc7, vl);
            
            // Advance output pointers by 8 columns
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
         * CRITICAL FIX: Handle tail columns (n % 8) using sequential sub-blocks
         * This fixes the bug where composite tails (3,5,6,7) were incorrectly handled
         * 
         * Process remaining columns in decreasing block sizes: 4, 2, 1
         * ------------------------------------------------------------------------ */
        
        // m8n4 loop - process 4-column blocks
        vl = vsetvl_e32m1(4);
        for (; j + 3 < n; j += 4) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            // Initialize accumulators for 4 columns
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            vfloat32m1_t _acc2 = vfmv_v_f_f32m1(bias_ptr[2], vl);
            vfloat32m1_t _acc3 = vfmv_v_f_f32m1(bias_ptr[3], vl);
            vfloat32m1_t _acc4 = vfmv_v_f_f32m1(bias_ptr[4], vl);
            vfloat32m1_t _acc5 = vfmv_v_f_f32m1(bias_ptr[5], vl);
            vfloat32m1_t _acc6 = vfmv_v_f_f32m1(bias_ptr[6], vl);
            vfloat32m1_t _acc7 = vfmv_v_f_f32m1(bias_ptr[7], vl);
            
            // Compute dot product
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 4;  // Move to next row in 4-block
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                float k4 = kernel_ptr[4];
                float k5 = kernel_ptr[5];
                float k6 = kernel_ptr[6];
                float k7 = kernel_ptr[7];
                kernel_ptr += 8;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f32m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f32m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f32m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f32m1(_acc7, k7, _input, vl);
            }
            
            // Store results
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            vse32_v_f32m1(out_ptr2, _acc2, vl);
            vse32_v_f32m1(out_ptr3, _acc3, vl);
            vse32_v_f32m1(out_ptr4, _acc4, vl);
            vse32_v_f32m1(out_ptr5, _acc5, vl);
            vse32_v_f32m1(out_ptr6, _acc6, vl);
            vse32_v_f32m1(out_ptr7, _acc7, vl);
            
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
        vl = vsetvl_e32m1(2);
        for (; j + 1 < n; j += 2) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            // Initialize accumulators for 2 columns
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            vfloat32m1_t _acc2 = vfmv_v_f_f32m1(bias_ptr[2], vl);
            vfloat32m1_t _acc3 = vfmv_v_f_f32m1(bias_ptr[3], vl);
            vfloat32m1_t _acc4 = vfmv_v_f_f32m1(bias_ptr[4], vl);
            vfloat32m1_t _acc5 = vfmv_v_f_f32m1(bias_ptr[5], vl);
            vfloat32m1_t _acc6 = vfmv_v_f_f32m1(bias_ptr[6], vl);
            vfloat32m1_t _acc7 = vfmv_v_f_f32m1(bias_ptr[7], vl);
            
            // Compute dot product
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 2;  // Move to next row in 2-block
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                float k4 = kernel_ptr[4];
                float k5 = kernel_ptr[5];
                float k6 = kernel_ptr[6];
                float k7 = kernel_ptr[7];
                kernel_ptr += 8;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f32m1(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f32m1(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f32m1(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f32m1(_acc7, k7, _input, vl);
            }
            
            // Store results
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            vse32_v_f32m1(out_ptr2, _acc2, vl);
            vse32_v_f32m1(out_ptr3, _acc3, vl);
            vse32_v_f32m1(out_ptr4, _acc4, vl);
            vse32_v_f32m1(out_ptr5, _acc5, vl);
            vse32_v_f32m1(out_ptr6, _acc6, vl);
            vse32_v_f32m1(out_ptr7, _acc7, vl);
            
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
        if (j < n) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            // Initialize scalar accumulators with bias
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[1];
            float acc2 = bias_ptr[2];
            float acc3 = bias_ptr[3];
            float acc4 = bias_ptr[4];
            float acc5 = bias_ptr[5];
            float acc6 = bias_ptr[6];
            float acc7 = bias_ptr[7];
            
            // Compute dot product using scalar operations
            for (int c = 0; c < k; c++) {
                float input_val = in_ptr[0];
                in_ptr += 1;  // Move to next row in 1-block
                
                acc0 += kernel_ptr[0] * input_val;
                acc1 += kernel_ptr[1] * input_val;
                acc2 += kernel_ptr[2] * input_val;
                acc3 += kernel_ptr[3] * input_val;
                acc4 += kernel_ptr[4] * input_val;
                acc5 += kernel_ptr[5] * input_val;
                acc6 += kernel_ptr[6] * input_val;
                acc7 += kernel_ptr[7] * input_val;
                
                kernel_ptr += 8;
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
        bias_ptr += 8;
        output_data += 8 * ldc;
        kernel_data += 8 * k;
    }

    /* ============================================================================
     * Process M4 blocks: Handle 4 rows at a time
     * Used when remaining rows >= 4 but < 8
     * ============================================================================ */
    for (; i + 3 < m; i += 4) {
        // Set up 4 output row pointers
        float *out_ptr0 = output_data;
        float *out_ptr1 = output_data + ldc;
        float *out_ptr2 = output_data + 2 * ldc;
        float *out_ptr3 = output_data + 3 * ldc;

        int j = 0;
        
        /* ------------------------------------------------------------------------
         * M4N8: Process 4x8 output tiles
         * ------------------------------------------------------------------------ */
        vl = vsetvl_e32m2(8);
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            // Initialize 4 accumulator vectors
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);

            // Compute dot product
            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                in_ptr += 8;
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                kernel_ptr += 4;

                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);
            }
            
            // Store results
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }
        
        /* ------------------------------------------------------------------------
         * Handle tail columns for M4 using sequential sub-blocks
         * ------------------------------------------------------------------------ */
        
        // m4n4 loop
        vl = vsetvl_e32m1(4);
        for (; j + 3 < n; j += 4) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            vfloat32m1_t _acc2 = vfmv_v_f_f32m1(bias_ptr[2], vl);
            vfloat32m1_t _acc3 = vfmv_v_f_f32m1(bias_ptr[3], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 4;
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                kernel_ptr += 4;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, k3, _input, vl);
            }
            
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            vse32_v_f32m1(out_ptr2, _acc2, vl);
            vse32_v_f32m1(out_ptr3, _acc3, vl);
            
            out_ptr0 += 4;
            out_ptr1 += 4;
            out_ptr2 += 4;
            out_ptr3 += 4;
        }
        
        // m4n2 loop
        vl = vsetvl_e32m1(2);
        for (; j + 1 < n; j += 2) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            vfloat32m1_t _acc2 = vfmv_v_f_f32m1(bias_ptr[2], vl);
            vfloat32m1_t _acc3 = vfmv_v_f_f32m1(bias_ptr[3], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 2;
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                kernel_ptr += 4;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, k3, _input, vl);
            }
            
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            vse32_v_f32m1(out_ptr2, _acc2, vl);
            vse32_v_f32m1(out_ptr3, _acc3, vl);
            
            out_ptr0 += 2;
            out_ptr1 += 2;
            out_ptr2 += 2;
            out_ptr3 += 2;
        }
        
        // m4n1 loop
        if (j < n) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[1];
            float acc2 = bias_ptr[2];
            float acc3 = bias_ptr[3];
            
            for (int c = 0; c < k; c++) {
                float input_val = in_ptr[0];
                in_ptr += 1;
                
                acc0 += kernel_ptr[0] * input_val;
                acc1 += kernel_ptr[1] * input_val;
                acc2 += kernel_ptr[2] * input_val;
                acc3 += kernel_ptr[3] * input_val;
                
                kernel_ptr += 4;
            }
            
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
            out_ptr2[0] = acc2;
            out_ptr3[0] = acc3;
        }
        
        // Advance to next block of 4 rows
        bias_ptr += 4;
        output_data += 4 * ldc;
        kernel_data += 4 * k;
    }

    /* ============================================================================
     * Process M2 blocks: Handle 2 rows at a time
     * Used when remaining rows >= 2 but < 4
     * ============================================================================ */
    for (; i + 1 < m; i += 2) {
        // Set up 2 output row pointers
        float *out_ptr0 = output_data;
        float *out_ptr1 = output_data + ldc;

        int j = 0;
        
        /* ------------------------------------------------------------------------
         * M2N8: Process 2x8 output tiles
         * ------------------------------------------------------------------------ */
        vl = vsetvl_e32m2(8);
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                in_ptr += 8;
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                kernel_ptr += 2;
                
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
            }
            
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            
            out_ptr0 += 8;
            out_ptr1 += 8;
        }

        /* ------------------------------------------------------------------------
         * Handle tail columns for M2 using sub-block approach
         * ------------------------------------------------------------------------ */
        
        // m2n4 loop
        vl = vsetvl_e32m1(4);
        for (; j + 3 < n; j += 4) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 4;
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                kernel_ptr += 2;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
            }
            
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            
            out_ptr0 += 4;
            out_ptr1 += 4;
        }
        
        // m2n2 loop
        vl = vsetvl_e32m1(2);
        for (; j + 1 < n; j += 2) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            vfloat32m1_t _acc1 = vfmv_v_f_f32m1(bias_ptr[1], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 2;
                
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                kernel_ptr += 2;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, k1, _input, vl);
            }
            
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr1, _acc1, vl);
            
            out_ptr0 += 2;
            out_ptr1 += 2;
        }
        
        // m2n1 loop
        if (j < n) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[1];
            
            for (int c = 0; c < k; c++) {
                float input_val = in_ptr[0];
                in_ptr += 1;
                
                acc0 += kernel_ptr[0] * input_val;
                acc1 += kernel_ptr[1] * input_val;
                
                kernel_ptr += 2;
            }
            
            out_ptr0[0] = acc0;
            out_ptr1[0] = acc1;
        }
        
        // Advance to next block of 2 rows
        bias_ptr += 2;
        output_data += 2 * ldc;
        kernel_data += 2 * k;
    }

    /* ============================================================================
     * Process M1 blocks: Handle last row (if any)
     * This handles the case when m is odd
     * ============================================================================ */
    for (; i < m; i++) {
        float *out_ptr0 = output_data;

        int j = 0;
        
        /* ------------------------------------------------------------------------
         * M1N8: Process 1x8 output tiles
         * ------------------------------------------------------------------------ */
        vl = vsetvl_e32m2(8);
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                in_ptr += 8;
                
                float k0 = kernel_ptr[0];
                kernel_ptr += 1;
                
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
            }
            
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            out_ptr0 += 8;
        }

        /* ------------------------------------------------------------------------
         * Handle tail columns for M1 using sub-block approach
         * ------------------------------------------------------------------------ */
        
        // m1n4 loop
        vl = vsetvl_e32m1(4);
        for (; j + 3 < n; j += 4) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 4;
                
                float k0 = kernel_ptr[0];
                kernel_ptr += 1;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
            }
            
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            out_ptr0 += 4;
        }
        
        // m1n2 loop
        vl = vsetvl_e32m1(2);
        for (; j + 1 < n; j += 2) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            vfloat32m1_t _acc0 = vfmv_v_f_f32m1(bias_ptr[0], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _input = vle32_v_f32m1(in_ptr, vl);
                in_ptr += 2;
                
                float k0 = kernel_ptr[0];
                kernel_ptr += 1;
                
                _acc0 = vfmacc_vf_f32m1(_acc0, k0, _input, vl);
            }
            
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            out_ptr0 += 2;
        }
        
        // m1n1 loop
        if (j < n) {
            float *kernel_ptr = kernel_data;
            float *in_ptr = input_data + sb_offset_z8(k, j);
            
            float acc = bias_ptr[0];
            
            for (int c = 0; c < k; c++) {
                acc += kernel_ptr[0] * in_ptr[0];
                kernel_ptr += 1;
                in_ptr += 1;
            }
            
            out_ptr0[0] = acc;
        }
        
        // Advance to next row
        bias_ptr++;
        output_data += ldc;
        kernel_data += k;
    }

    // Clean up: free temporary bias if we allocated it
    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
