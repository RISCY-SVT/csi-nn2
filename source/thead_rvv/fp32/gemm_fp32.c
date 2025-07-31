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

/************************************************************************************
 * GEMM operation: C = A * B + bias
 * where:
 *   C (dst) - output matrix [m, n]
 *   A (sa)  - kernel matrix [m, k] - reordered by shl_rvv_reorder_kernel_n8_fp32
 *   B (sb)  - input matrix  [k, n] - reordered by shl_rvv_reorder_input_z8_fp32
 *   bias    - bias vector [m] (optional)
 *
 * ===================================================================================
 * CRITICAL: Data layout after reordering (HYBRID FORMAT)
 * ===================================================================================
 * 
 * 1. Kernel matrix (A) format after reorder_kernel_n8:
 *    - Column-major within 8×k blocks
 *    - Each block stores 8 rows × k columns as [col0][col1]...[col(k-1)]
 *    - Each column contains 8 consecutive elements from different rows
 *    Example for 8×3 matrix:
 *      Original: a00 a01 a02    Reordered: [a00 a10 a20...a70] [a01 a11 a21...a71] [a02 a12 a22...a72]
 *                a10 a11 a12               └─── column 0 ───┘ └─── column 1 ───┘ └─── column 2 ───┘
 *                ...
 *                a70 a71 a72
 *
 * 2. Input matrix (B) format after reorder_input_z8 (HYBRID):
 *    - Full 8-column blocks: ROW-MAJOR format
 *      * Stored as [row0][row1]...[row(k-1)]
 *      * Each row contains 8 consecutive elements
 *      * Enables efficient vector loads
 *    
 *    - Tail blocks (< 8 columns): COLUMN-MAJOR format
 *      * Stored as [col0][col1]...[col(n%8-1)]
 *      * Each column contains k consecutive elements
 *      * Simplifies pointer arithmetic for small tiles
 *
 *    Example for k=2, n=11 matrix:
 *      Original: b00 b01 b02 ... b08 b09 b0,10
 *                b10 b11 b12 ... b18 b19 b1,10
 *      
 *      Reordered: [b00 b01...b07][b10 b11...b17] | [b08 b18] [b09 b19] [b0,10 b1,10]
 *                 └─ 8-block row-major ─────────┘   └─ 3-tail column-major ─────────┘
 *
 * ===================================================================================
 * Processing strategy:
 * - Outer loop: Process m in blocks of 8, 4, 2, 1
 * - Inner loop: Process n in blocks of 8, 4, 2, 1
 * - Full 8-blocks use vector loads/stores
 * - Tail blocks use appropriate access patterns based on format
 * - Strided stores (vsse) used for column-major output
 ************************************************************************************/

void shl_rvv_gemm_8x8_fp32(float *dst, const float *sa, const float *sb, float *bias, int m, int k,
                           int n, int ldc)
{
    float *kernel_data = (float *)sa;
    float *input_data = (float *)sb;
    float *output_data = dst;

    int flag_bias = 1;  // default: conv2d layer include bias
    if (bias == NULL) {
        flag_bias = 0;
        bias = (float *)shl_mem_alloc(m * sizeof(float));
    }
    float *bias_ptr = bias;

    int vl;

    int i = 0;
    // m8 loop - process 8 output rows at a time
    vl = vsetvl_e32m2(8);
    for (; i + 7 < m; i += 8) {
        float *in_ptr = input_data;

        // Set up output row pointers
        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;
        float *out_ptr2 = out_ptr1 + ldc;
        float *out_ptr3 = out_ptr2 + ldc;
        float *out_ptr4 = out_ptr3 + ldc;
        float *out_ptr5 = out_ptr4 + ldc;
        float *out_ptr6 = out_ptr5 + ldc;
        float *out_ptr7 = out_ptr6 + ldc;

        int j = 0;
        
        // Process full 8-column blocks (ROW-MAJOR format in input)
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            
            // Initialize accumulators with bias
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);
            vfloat32m2_t _acc4 = vfmv_v_f_f32m2(bias_ptr[4], vl);
            vfloat32m2_t _acc5 = vfmv_v_f_f32m2(bias_ptr[5], vl);
            vfloat32m2_t _acc6 = vfmv_v_f_f32m2(bias_ptr[6], vl);
            vfloat32m2_t _acc7 = vfmv_v_f_f32m2(bias_ptr[7], vl);

            // Main GEMM computation
            for (int c = 0; c < k; c++) {
                // Load 8 elements from current row of B (row-major)
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);

                // Load 8 elements from current column of A (column-major within block)
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];
                float k4 = kernel_ptr[4];
                float k5 = kernel_ptr[5];
                float k6 = kernel_ptr[6];
                float k7 = kernel_ptr[7];

                // Multiply and accumulate
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);
                _acc4 = vfmacc_vf_f32m2(_acc4, k4, _input, vl);
                _acc5 = vfmacc_vf_f32m2(_acc5, k5, _input, vl);
                _acc6 = vfmacc_vf_f32m2(_acc6, k6, _input, vl);
                _acc7 = vfmacc_vf_f32m2(_acc7, k7, _input, vl);

                kernel_ptr += 8;  // Next column of A (8 elements)
                in_ptr += 8;      // Next row of B (8 elements in row-major)
            }
            
            // Store results
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            vse32_v_f32m2(out_ptr4, _acc4, vl);
            vse32_v_f32m2(out_ptr5, _acc5, vl);
            vse32_v_f32m2(out_ptr6, _acc6, vl);
            vse32_v_f32m2(out_ptr7, _acc7, vl);
            
            // Advance output pointers
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
            out_ptr4 += 8;
            out_ptr5 += 8;
            out_ptr6 += 8;
            out_ptr7 += 8;
        }
        
        // Process tail columns (COLUMN-MAJOR format in input for blocks < 8)
        // m8n4 - 4 columns
        for (; j + 3 < n; j += 4) {
            // Initialize accumulators with bias (broadcast to vector)
            vfloat32m2_t _acc0 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc1 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc2 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc3 = vle32_v_f32m2(bias_ptr, vl);

            float *kernel_ptr = kernel_data;

            // Set up column pointers for column-major input
            float *in_ptr0 = in_ptr;      // Column 0
            float *in_ptr1 = in_ptr0 + k;  // Column 1
            float *in_ptr2 = in_ptr1 + k;  // Column 2
            float *in_ptr3 = in_ptr2 + k;  // Column 3

            // Set up strided output pointers
            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            // Compute 8x4 block
            for (int c = 0; c < k; c++) {
                // Load kernel column (8 elements)
                vfloat32m2_t _kernel = vle32_v_f32m2(kernel_ptr, vl);
                
                // Multiply kernel column by each input scalar and accumulate
                _acc0 = vfmacc_vf_f32m2(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, in_ptr3[c], _kernel, vl);
                
                kernel_ptr += 8;
            }
            
            // Store with stride to maintain row-major output
            vsse32_v_f32m2(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m2(out_ptr1, ldc * sizeof(float), _acc1, vl);
            vsse32_v_f32m2(out_ptr2, ldc * sizeof(float), _acc2, vl);
            vsse32_v_f32m2(out_ptr3, ldc * sizeof(float), _acc3, vl);
            
            out_ptr0 += 4;
            in_ptr += 4 * k;  // Skip 4 columns
        }
        
        // m8n2 - 2 columns
        for (; j + 1 < n; j += 2) {
            vfloat32m2_t _acc0 = vle32_v_f32m2(bias_ptr, vl);
            vfloat32m2_t _acc1 = vle32_v_f32m2(bias_ptr, vl);

            float *kernel_ptr = kernel_data;

            // Column-major input pointers
            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;

            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _kernel = vle32_v_f32m2(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 8;
            }
            
            vsse32_v_f32m2(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m2(out_ptr1, ldc * sizeof(float), _acc1, vl);
            
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        
        // m8n1 - 1 column
        for (; j < n; j++) {
            vfloat32m2_t _acc0 = vle32_v_f32m2(bias_ptr, vl);
            float *kernel_ptr = kernel_data;
            float *in_ptr0 = in_ptr;

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _kernel = vle32_v_f32m2(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m2(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 8;
            }
            
            vsse32_v_f32m2(out_ptr0, ldc * sizeof(float), _acc0, vl);
            out_ptr0 += 1;
            in_ptr += k;
        }
        
        kernel_data += 8 * k;
        output_data += 8 * ldc;
        bias_ptr += 8;
    }

    // m4 loop - process 4 output rows at a time
    for (; i + 3 < m; i += 4) {
        vl = vsetvl_e32m2(8);  // Still use vl=8 for n dimension
        float *in_ptr = input_data;

        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;
        float *out_ptr2 = out_ptr1 + ldc;
        float *out_ptr3 = out_ptr2 + ldc;

        int j = 0;
        
        // m4n8 - process 8-column blocks (row-major)
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);
            vfloat32m2_t _acc2 = vfmv_v_f_f32m2(bias_ptr[2], vl);
            vfloat32m2_t _acc3 = vfmv_v_f_f32m2(bias_ptr[3], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                float k2 = kernel_ptr[2];
                float k3 = kernel_ptr[3];

                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                _acc2 = vfmacc_vf_f32m2(_acc2, k2, _input, vl);
                _acc3 = vfmacc_vf_f32m2(_acc3, k3, _input, vl);

                kernel_ptr += 4;
                in_ptr += 8;
            }
            
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            vse32_v_f32m2(out_ptr2, _acc2, vl);
            vse32_v_f32m2(out_ptr3, _acc3, vl);
            
            out_ptr0 += 8;
            out_ptr1 += 8;
            out_ptr2 += 8;
            out_ptr3 += 8;
        }
        
        // Switch to smaller vector length for tail processing
        vl = vsetvl_e32m1(4);
        
        // m4n4 - column-major tail
        for (; j + 3 < n; j += 4) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc1 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc2 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc3 = vle32_v_f32m1(bias_ptr, vl);

            float *kernel_ptr = kernel_data;

            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            float *in_ptr2 = in_ptr1 + k;
            float *in_ptr3 = in_ptr2 + k;

            out_ptr1 = out_ptr0 + 1;
            out_ptr2 = out_ptr0 + 2;
            out_ptr3 = out_ptr0 + 3;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 4;
            }
            
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m1(out_ptr1, ldc * sizeof(float), _acc1, vl);
            vsse32_v_f32m1(out_ptr2, ldc * sizeof(float), _acc2, vl);
            vsse32_v_f32m1(out_ptr3, ldc * sizeof(float), _acc3, vl);
            
            out_ptr0 += 4;
            in_ptr += 4 * k;
        }
        
        // m4n2 - column-major tail
        for (; j + 1 < n; j += 2) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc1 = vle32_v_f32m1(bias_ptr, vl);

            float *kernel_ptr = kernel_data;
            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            out_ptr1 = out_ptr0 + 1;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, in_ptr1[c], _kernel, vl);
                kernel_ptr += 4;
            }
            
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
            vsse32_v_f32m1(out_ptr1, ldc * sizeof(float), _acc1, vl);
            
            out_ptr0 += 2;
            in_ptr += 2 * k;
        }
        
        // m4n1 - column-major tail
        for (; j < n; j++) {
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            float *kernel_ptr = kernel_data;
            float *in_ptr0 = in_ptr;

            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                kernel_ptr += 4;
            }
            
            vsse32_v_f32m1(out_ptr0, ldc * sizeof(float), _acc0, vl);
            out_ptr0 += 1;
            in_ptr += k;
        }
        
        kernel_data += 4 * k;
        output_data += 4 * ldc;
        bias_ptr += 4;
    }

    // m2 loop - process 2 output rows at a time
    for (; i + 1 < m; i += 2) {
        vl = vsetvl_e32m2(8);
        float *in_ptr = input_data;
        float *out_ptr0 = output_data;
        float *out_ptr1 = out_ptr0 + ldc;

        int j = 0;
        
        // m2n8 - process 8-column blocks (row-major)
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            vfloat32m2_t _acc1 = vfmv_v_f_f32m2(bias_ptr[1], vl);

            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                float k0 = kernel_ptr[0];
                float k1 = kernel_ptr[1];
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                _acc1 = vfmacc_vf_f32m2(_acc1, k1, _input, vl);
                kernel_ptr += 2;
                in_ptr += 8;
            }
            
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            vse32_v_f32m2(out_ptr1, _acc1, vl);
            
            out_ptr0 += 8;
            out_ptr1 += 8;
        }

        // Process tail with vectorization where possible
        vl = vsetvl_e32m1(2);  // Process 2 rows at once
        
        // m2n4 - vectorized column-major tail
        for (; j + 3 < n; j += 4) {
            float *kernel_ptr = kernel_data;
            
            // Set up column pointers
            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            float *in_ptr2 = in_ptr1 + k;
            float *in_ptr3 = in_ptr2 + k;
            
            vfloat32m1_t _acc0 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc1 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc2 = vle32_v_f32m1(bias_ptr, vl);
            vfloat32m1_t _acc3 = vle32_v_f32m1(bias_ptr, vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m1_t _kernel = vle32_v_f32m1(kernel_ptr, vl);
                _acc0 = vfmacc_vf_f32m1(_acc0, in_ptr0[c], _kernel, vl);
                _acc1 = vfmacc_vf_f32m1(_acc1, in_ptr1[c], _kernel, vl);
                _acc2 = vfmacc_vf_f32m1(_acc2, in_ptr2[c], _kernel, vl);
                _acc3 = vfmacc_vf_f32m1(_acc3, in_ptr3[c], _kernel, vl);
                kernel_ptr += 2;
            }
            
            // Store results [out_ptr0[0], out_ptr1[0]] for column 0
            vse32_v_f32m1(out_ptr0, _acc0, vl);
            vse32_v_f32m1(out_ptr0 + 1, _acc1, vl);
            vse32_v_f32m1(out_ptr0 + 2, _acc2, vl);
            vse32_v_f32m1(out_ptr0 + 3, _acc3, vl);
            
            out_ptr0 += 4;
            out_ptr1 += 4;
            in_ptr += 4 * k;
        }
        
        // m2n2 and m2n1 - fall back to scalar for simplicity
        for (; j < n; j++) {
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[1];
            
            for (int c = 0; c < k; c++) {
                // kernel is column-major within blocks
                acc0 += kernel_data[2 * c] * in_ptr[c];
                acc1 += kernel_data[2 * c + 1] * in_ptr[c];
            }
            
            *out_ptr0++ = acc0;
            *out_ptr1++ = acc1;
            in_ptr += k;
        }
        
        kernel_data += 2 * k;
        output_data += 2 * ldc;
        bias_ptr += 2;
    }

    // m1 loop - process 1 output row at a time
    for (; i < m; i++) {
        vl = vsetvl_e32m2(8);
        float *in_ptr = input_data;
        float *out_ptr0 = output_data;

        int j = 0;
        
        // m1n8 - process 8-column blocks (row-major)
        for (; j + 7 < n; j += 8) {
            float *kernel_ptr = kernel_data;
            vfloat32m2_t _acc0 = vfmv_v_f_f32m2(bias_ptr[0], vl);
            
            for (int c = 0; c < k; c++) {
                vfloat32m2_t _input = vle32_v_f32m2(in_ptr, vl);
                float k0 = kernel_ptr[0];
                _acc0 = vfmacc_vf_f32m2(_acc0, k0, _input, vl);
                kernel_ptr += 1;
                in_ptr += 8;
            }
            
            vse32_v_f32m2(out_ptr0, _acc0, vl);
            out_ptr0 += 8;
        }

        // m1 tail - vectorized where beneficial
        if (j + 3 < n) {
            // Process 4 columns at once
            vl = vsetvl_e32m1(4);
            float *kernel_ptr = kernel_data;
            
            // Column pointers
            float *in_ptr0 = in_ptr;
            float *in_ptr1 = in_ptr0 + k;
            float *in_ptr2 = in_ptr1 + k;
            float *in_ptr3 = in_ptr2 + k;
            
            // Use scalar accumulators for single row
            float acc0 = bias_ptr[0];
            float acc1 = bias_ptr[0];
            float acc2 = bias_ptr[0];
            float acc3 = bias_ptr[0];
            
            for (int c = 0; c < k; c++) {
                float k_val = kernel_ptr[c];
                acc0 += k_val * in_ptr0[c];
                acc1 += k_val * in_ptr1[c];
                acc2 += k_val * in_ptr2[c];
                acc3 += k_val * in_ptr3[c];
            }
            
            out_ptr0[0] = acc0;
            out_ptr0[1] = acc1;
            out_ptr0[2] = acc2;
            out_ptr0[3] = acc3;
            
            out_ptr0 += 4;
            in_ptr += 4 * k;
            j += 4;
        }
        
        // Process remaining columns
        for (; j < n; j++) {
            float acc0 = bias_ptr[0];
            
            for (int c = 0; c < k; c++) {
                acc0 += kernel_data[c] * in_ptr[c];
            }
            
            *out_ptr0++ = acc0;
            in_ptr += k;
        }
        
        kernel_data += k;
        output_data += ldc;
        bias_ptr += 1;
    }

    if (!flag_bias) {
        shl_mem_free(bias);
        bias = NULL;
    }
}
