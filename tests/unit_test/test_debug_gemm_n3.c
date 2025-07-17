/*
 * Debug test specifically for n=3 case in GEMM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csi_nn.h"
#include "shl_utils.h"
#include "rvv/rvv.h"

// External functions
extern void shl_rvv_gemm_8x16_fp16(__fp16 *dst, const __fp16 *sa, const __fp16 *sb, 
                                   __fp16 *bias, int m, int k, int n, int ldc);
extern void shl_rvv_reorder_kernel_n8_fp16(__fp16 *a, __fp16 *sa, int m, int k, int ldx);
extern void shl_rvv_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);

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

// Print reordered buffer structure
void print_reordered_buffer(const char *name, __fp16 *buf, int k, int n) {
    printf("\n%s reordered buffer structure (k=%d, n=%d):\n", name, k, n);
    
    // For n=3, after reorder_input_z16, we expect:
    // - No 16-block (n < 16)
    // - No 8-block (n < 8)  
    // - No 4-block (n < 4)
    // - One 2-block for columns 0-1
    // - One 1-block for column 2
    
    if (n == 3) {
        printf("Expected layout: [2-block: k*2 elements] [1-block: k*1 elements]\n");
        printf("2-block (columns 0-1):\n");
        for (int row = 0; row < k && row < 5; row++) {
            printf("  row %d: ", row);
            for (int col = 0; col < 2; col++) {
                printf("%6.3f ", (float)buf[row * 2 + col]);
            }
            printf("\n");
        }
        if (k > 5) printf("  ...\n");
        
        printf("1-block (column 2):\n");
        for (int row = 0; row < k && row < 5; row++) {
            printf("  row %d: %6.3f\n", row, (float)buf[k * 2 + row]);
        }
        if (k > 5) printf("  ...\n");
    }
}

// Test helper function offset calculation
size_t sb_offset_z16_test(size_t k, int col) {
    size_t off = 0;
    printf("  sb_offset_z16(k=%zu, col=%d): ", k, col);
    
    // Process full 16-column blocks
    if (col >= 16) {
        int blk16 = col / 16;
        off += (size_t)blk16 * k * 16;
        col -= blk16 * 16;
        printf("16-blk:%d ", blk16);
    }
    
    // Process remaining columns in decreasing block sizes
    if (col >= 8) { 
        off += k * 8; 
        col -= 8; 
        printf("8-blk ");
    }
    if (col >= 4) { 
        off += k * 4; 
        col -= 4; 
        printf("4-blk ");
    }
    if (col >= 2) { 
        off += k * 2; 
        col -= 2; 
        printf("2-blk ");
    }
    if (col >= 1) { 
        off += k * 1; 
        col -= 1; 
        printf("1-blk ");
    }
    
    printf("-> offset=%zu\n", off);
    return off;
}

int main() {
    printf("Debug test for GEMM n=3 case\n");
    printf("=============================\n");
    
    // Test parameters: m=4, k=3, n=3 (simple case)
    const int m = 4, k = 3, n = 3;
    
    // Allocate matrices
    __fp16 *A = (__fp16*)shl_mem_alloc(m * k * sizeof(__fp16));
    __fp16 *B = (__fp16*)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *A_reordered = (__fp16*)shl_mem_alloc(m * k * sizeof(__fp16));
    __fp16 *B_reordered = (__fp16*)shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *C_out = (__fp16*)shl_mem_alloc(m * n * sizeof(__fp16));
    __fp16 *C_ref = (__fp16*)shl_mem_alloc(m * n * sizeof(__fp16));
    
    // Initialize with simple sequential values
    printf("\nInitializing matrices:\n");
    for (int i = 0; i < m * k; i++) {
        A[i] = (__fp16)(i * 0.1f);
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (__fp16)(i * 0.1f);
    }
    
    // Print original B matrix
    printf("\nOriginal B matrix (k=%d, n=%d):\n", k, n);
    for (int i = 0; i < k; i++) {
        printf("  row %d: ", i);
        for (int j = 0; j < n; j++) {
            printf("%6.3f ", (float)B[i * n + j]);
        }
        printf("\n");
    }
    
    // Test offset calculation
    printf("\nTesting offset calculations for n=3:\n");
    for (int col = 0; col < n; col++) {
        sb_offset_z16_test(k, col);
    }
    
    // Reorder matrices
    printf("\nReordering matrices...\n");
    shl_rvv_reorder_kernel_n8_fp16(A, A_reordered, m, k, k);
    shl_rvv_reorder_input_z16_fp16(B, B_reordered, k, n, n);
    
    // Print reordered B structure
    print_reordered_buffer("B", B_reordered, k, n);
    
    // Clear output
    memset(C_out, 0, m * n * sizeof(__fp16));
    memset(C_ref, 0, m * n * sizeof(__fp16));
    
    // Compute reference
    printf("\nComputing reference result...\n");
    ref_gemm_fp16(C_ref, A, B, NULL, m, k, n, n);
    
    // Call RVV GEMM
    printf("\nCalling RVV GEMM...\n");
    shl_rvv_gemm_8x16_fp16(C_out, A_reordered, B_reordered, NULL, m, k, n, n);
    
    // Compare results
    printf("\nResults comparison:\n");
    printf("       Reference         RVV Output       Difference\n");
    int errors = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float ref = (float)C_ref[i * n + j];
            float out = (float)C_out[i * n + j];
            float diff = fabsf(ref - out);
            printf("[%d,%d]: %8.5f  vs  %8.5f  diff=%8.5f %s\n", 
                   i, j, ref, out, diff, 
                   diff > 0.001f ? "ERROR" : "OK");
            if (diff > 0.001f) errors++;
        }
    }
    
    printf("\nTotal errors: %d/%d\n", errors, m * n);
    
    // Manual verification of one element
    printf("\nManual calculation for C[0,0]:\n");
    float manual_sum = 0.0f;
    for (int p = 0; p < k; p++) {
        float a_val = (float)A[0 * k + p];
        float b_val = (float)B[p * n + 0];
        manual_sum += a_val * b_val;
        printf("  A[0,%d] * B[%d,0] = %.3f * %.3f = %.3f\n", 
               p, p, a_val, b_val, a_val * b_val);
    }
    printf("  Sum = %.5f\n", manual_sum);
    
    // Cleanup
    shl_mem_free(A);
    shl_mem_free(B);
    shl_mem_free(A_reordered);
    shl_mem_free(B_reordered);
    shl_mem_free(C_out);
    shl_mem_free(C_ref);
    
    return errors > 0 ? 1 : 0;
}
