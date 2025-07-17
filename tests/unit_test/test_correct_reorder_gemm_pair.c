/*
 * Test using the correct pair of reorder/gemm functions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csi_nn.h"
#include "shl_utils.h"
#include "rvv/rvv.h"

// External functions - try different combinations
extern void shl_rvv_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, 
                                   __fp16 *bias, int m, int k, int n, int ldc);
extern void shl_rvv_reorder_kernel_n8_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
extern void shl_rvv_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);
extern void shl_rvv_reorder_input_z8_packn_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);

// Simple reference GEMM
void ref_gemm_fp16(__fp16 *C, const __fp16 *A, const __fp16 *B, const __fp16 *bias,
                   int m, int k, int n, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            __fp16 sum = bias ? bias[i] : 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

int test_gemm_with_reorder(const char *name, 
                           void (*reorder_func)(__fp16*, __fp16*, int, int, int),
                           int m, int k, int n) {
    printf("\n=== Testing %s: m=%d, k=%d, n=%d ===\n", name, m, k, n);
    
    // Allocate matrices
    __fp16 *A = (__fp16*)shl_mem_alloc(m * k * sizeof(__fp16));
    __fp16 *B = (__fp16*)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *A_reordered = (__fp16*)shl_mem_alloc(m * k * sizeof(__fp16));
    __fp16 *B_reordered = (__fp16*)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *C_out = (__fp16*)shl_mem_alloc(m * n * sizeof(__fp16));
    __fp16 *C_ref = (__fp16*)shl_mem_alloc(m * n * sizeof(__fp16));
    
    // Initialize
    for (int i = 0; i < m * k; i++) {
        A[i] = (__fp16)((i % 10) * 0.1f);
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (__fp16)((i % 10) * 0.1f);
    }
    
    // Compute reference
    memset(C_ref, 0, m * n * sizeof(__fp16));
    ref_gemm_fp16(C_ref, A, B, NULL, m, k, n, n);
    
    // Reorder and compute
    shl_rvv_reorder_kernel_n8_fp16(A, A_reordered, m, k, k);
    reorder_func(B, B_reordered, k, n, n);
    
    memset(C_out, 0, m * n * sizeof(__fp16));
    shl_rvv_gemm_8x16_fp16(C_out, A_reordered, B_reordered, NULL, m, k, n, n);
    
    // Compare
    int errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < m * n; i++) {
        float diff = fabsf((float)C_out[i] - (float)C_ref[i]);
        if (diff > 0.01f) {
            if (errors < 5) {
                printf("  Error at [%d]: ref=%.4f, out=%.4f, diff=%.4f\n",
                       i, (float)C_ref[i], (float)C_out[i], diff);
            }
            errors++;
        }
        if (diff > max_error) max_error = diff;
    }
    
    printf("  Errors: %d/%d, Max error: %.6f\n", errors, m * n, max_error);
    
    // Cleanup
    shl_mem_free(A);
    shl_mem_free(B);
    shl_mem_free(A_reordered);
    shl_mem_free(B_reordered);
    shl_mem_free(C_out);
    shl_mem_free(C_ref);
    
    return errors;
}

int main() {
    printf("Testing different reorder/GEMM function combinations\n");
    printf("===================================================\n");
    
    // Test cases with different n values (especially those that fail)
    int test_configs[][3] = {
        {4, 3, 3},   // n=3 case
        {4, 3, 5},   // n=5 case
        {4, 3, 7},   // n=7 case
        {4, 3, 8},   // n=8 (should work)
        {4, 3, 9},   // n=9 case
    };
    
    int total_errors = 0;
    
    // Test with reorder_input_z16
    printf("\n--- Using shl_rvv_reorder_input_z16_fp16 ---\n");
    for (int i = 0; i < 5; i++) {
        total_errors += test_gemm_with_reorder("z16", shl_rvv_reorder_input_z16_fp16,
                                               test_configs[i][0], test_configs[i][1], 
                                               test_configs[i][2]);
    }
    
    // Test with reorder_input_z8_packn (if available)
    printf("\n--- Using shl_rvv_reorder_input_z8_packn_fp16 ---\n");
    for (int i = 0; i < 5; i++) {
        total_errors += test_gemm_with_reorder("z8_packn", shl_rvv_reorder_input_z8_packn_fp16,
                                               test_configs[i][0], test_configs[i][1], 
                                               test_configs[i][2]);
    }
    
    printf("\n===================================================\n");
    printf("Total errors: %d\n", total_errors);
    
    return total_errors > 0 ? 1 : 0;
}
