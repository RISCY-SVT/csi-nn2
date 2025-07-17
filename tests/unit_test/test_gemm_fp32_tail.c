/*
 * test_gemm_fp32_tail.c - Test FP32 GEMM tail cases after fix
 * 
 * Build:
 *   cd /data/csi-nn2
 *   riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv0p7_zfh_xtheadc_xtheadvdot \
 *     -mabi=lp64d -mtune=c920 -I./include -I. \
 *     -Wall -Wextra -Werror \
 *     test_gemm_fp32_tail.c \
 *     source/thead_rvv/fp32/gemm_fp32.c \
 *     source/thead_rvv/reorder.c \
 *     source/utils/debug.c \
 *     -lm -o test_gemm_fp32_tail.elf
 * 
 * Run:
 *   qemu-riscv64 -cpu rv64,x-v=true,vlen=128,x-zvfh=true ./test_gemm_fp32_tail.elf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "backend/rvv/rvv.h"
#include "csinn/csinn_data_structure.h"
#include "shl_utils.h"

#define COLOR_RED    "\x1b[31m"
#define COLOR_GREEN  "\x1b[32m"
#define COLOR_RESET  "\x1b[0m"

// Simple reference GEMM
static void ref_gemm_fp32(float *C, const float *A, const float *B,
                          const float *bias, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = bias ? bias[i] : 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Test one configuration
static int test_config(int m, int n, int k) {
    printf("Testing m=%d, n=%d, k=%d ... ", m, n, k);
    
    // Allocate matrices
    float *A = (float*)shl_mem_alloc(m * k * sizeof(float));
    float *B = (float*)shl_mem_alloc(k * n * sizeof(float));
    float *A_reordered = (float*)shl_mem_alloc(m * k * sizeof(float));
    float *B_reordered = (float*)shl_mem_alloc(k * n * sizeof(float));
    float *C_out = (float*)shl_mem_alloc(m * n * sizeof(float));
    float *C_ref = (float*)shl_mem_alloc(m * n * sizeof(float));
    float *bias = (float*)shl_mem_alloc(m * sizeof(float));
    
    // Initialize with simple pattern
    for (int i = 0; i < m * k; i++) A[i] = (float)(i % 7 - 3) * 0.1f;
    for (int i = 0; i < k * n; i++) B[i] = (float)(i % 5 - 2) * 0.2f;
    for (int i = 0; i < m; i++) bias[i] = (float)(i % 3) * 0.5f;
    
    // Clear outputs
    memset(C_out, 0, m * n * sizeof(float));
    memset(C_ref, 0, m * n * sizeof(float));
    
    // Reference implementation
    ref_gemm_fp32(C_ref, A, B, bias, m, k, n);
    
    // Reorder for RVV
    shl_rvv_reorder_kernel_n8_fp32(A, A_reordered, m, k, k);
    shl_rvv_reorder_input_z8_fp32(B, B_reordered, k, n, n);
    
    // Run RVV GEMM
    shl_rvv_gemm_8x8_fp32(C_out, A_reordered, B_reordered, bias, m, k, n, n);
    
    // Compare results
    int errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float ref_val = C_ref[i * n + j];
            float out_val = C_out[i * n + j];
            float error = fabsf(out_val - ref_val);
            if (error > max_error) max_error = error;
            if (error > 1e-4f) {
                if (errors < 5) {
                    printf("\n  [%d,%d]: out=%.6f ref=%.6f diff=%.6e",
                           i, j, out_val, ref_val, error);
                }
                errors++;
            }
        }
    }
    
    // Cleanup
    shl_mem_free(A);
    shl_mem_free(B);
    shl_mem_free(A_reordered);
    shl_mem_free(B_reordered);
    shl_mem_free(C_out);
    shl_mem_free(C_ref);
    shl_mem_free(bias);
    
    if (errors == 0) {
        printf(COLOR_GREEN "PASS" COLOR_RESET " (max_error=%.2e)\n", max_error);
        return 0;
    } else {
        printf(COLOR_RED "FAIL" COLOR_RESET " (%d errors, max_error=%.2e)\n", 
               errors, max_error);
        return 1;
    }
}

int main() {
    printf("=== FP32 GEMM Tail Test ===\n\n");
    
    int total_errors = 0;
    
    // Test various tail cases
    printf("Tail-N cases (n %% 8 != 0):\n");
    total_errors += test_config(8, 3, 4);   // n=3 (needs 2+1)
    total_errors += test_config(8, 5, 4);   // n=5 (needs 4+1)
    total_errors += test_config(8, 6, 4);   // n=6 (needs 4+2)
    total_errors += test_config(8, 7, 4);   // n=7 (needs 4+2+1)
    total_errors += test_config(8, 9, 4);   // n=9 (8+1)
    total_errors += test_config(8, 11, 4);  // n=11 (8+2+1)
    total_errors += test_config(8, 13, 4);  // n=13 (8+4+1)
    total_errors += test_config(8, 15, 4);  // n=15 (8+4+2+1)
    
    printf("\nTail-M cases (m %% 8 != 0):\n");
    total_errors += test_config(3, 8, 4);   // m=3
    total_errors += test_config(5, 8, 4);   // m=5 (4+1)
    total_errors += test_config(6, 8, 4);   // m=6 (4+2)
    total_errors += test_config(7, 8, 4);   // m=7 (4+2+1)
    
    printf("\nMixed tail cases:\n");
    total_errors += test_config(3, 5, 7);   // m=3, n=5
    total_errors += test_config(7, 11, 5);  // m=7, n=11
    total_errors += test_config(15, 23, 9); // larger case
    
    printf("\nEdge cases:\n");
    total_errors += test_config(1, 1, 1);   // minimal
    total_errors += test_config(1, 17, 3);  // single row
    total_errors += test_config(17, 1, 3);  // single column
    
    printf("\n=== Summary ===\n");
    if (total_errors == 0) {
        printf(COLOR_GREEN "All tests passed!" COLOR_RESET "\n");
    } else {
        printf(COLOR_RED "%d tests failed!" COLOR_RESET "\n", total_errors);
    }
    
    return total_errors;
}
