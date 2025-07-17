/*
 * Test program to verify RVV instruction behavior for tail processing patterns
 * This tests the specific vector operations used in GEMM tail handling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rvv/rvv.h"

// Test 1: Basic vector load/store with different LMUL and element counts
void test_basic_vle_vse() {
    printf("\n=== Test 1: Basic VLE/VSE with different LMUL ===\n");
    
    __fp16 src[32], dst[32];
    for (int i = 0; i < 32; i++) {
        src[i] = (__fp16)(i + 0.5f);
        dst[i] = 0;
    }
    
    // Test LMUL=2 with vl=16
    int vl = vsetvl_e16m2(16);
    printf("vsetvl_e16m2(16) returned vl=%d (expected 16)\n", vl);
    vfloat16m2_t v1 = vle16_v_f16m2(src, vl);
    vse16_v_f16m2(dst, v1, vl);
    
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        if (fabsf((float)dst[i] - (float)src[i]) > 0.001f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, (float)src[i], (float)dst[i]);
            errors++;
        }
    }
    printf("LMUL=2, vl=16: %s\n", errors ? "FAILED" : "PASSED");
    
    // Reset dst
    memset(dst, 0, sizeof(dst));
    
    // Test LMUL=1 with vl=8
    vl = vsetvl_e16m1(8);
    printf("\nvsetvl_e16m1(8) returned vl=%d (expected 8)\n", vl);
    vfloat16m1_t v2 = vle16_v_f16m1(src, vl);
    vse16_v_f16m1(dst, v2, vl);
    
    errors = 0;
    for (int i = 0; i < 8; i++) {
        if (fabsf((float)dst[i] - (float)src[i]) > 0.001f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, (float)src[i], (float)dst[i]);
            errors++;
        }
    }
    printf("LMUL=1, vl=8: %s\n", errors ? "FAILED" : "PASSED");
    
    // Test LMUL=1 with vl=4
    memset(dst, 0, sizeof(dst));
    vl = vsetvl_e16m1(4);
    printf("\nvsetvl_e16m1(4) returned vl=%d (expected 4)\n", vl);
    v2 = vle16_v_f16m1(src, vl);
    vse16_v_f16m1(dst, v2, vl);
    
    errors = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf((float)dst[i] - (float)src[i]) > 0.001f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, (float)src[i], (float)dst[i]);
            errors++;
        }
    }
    printf("LMUL=1, vl=4: %s\n", errors ? "FAILED" : "PASSED");
}

// Test 2: Vector FMACC operations with different vl
void test_vfmacc_patterns() {
    printf("\n=== Test 2: VFMACC patterns ===\n");
    
    __fp16 input[16], result[16];
    for (int i = 0; i < 16; i++) {
        input[i] = (__fp16)(i + 1);
    }
    
    // Test LMUL=2, vl=16
    int vl = vsetvl_e16m2(16);
    vfloat16m2_t vinput = vle16_v_f16m2(input, vl);
    vfloat16m2_t vacc = vfmv_v_f_f16m2((__fp16)0.0, vl);
    
    // Perform FMACC: acc = acc + 2.0 * input
    vacc = vfmacc_vf_f16m2(vacc, (__fp16)2.0, vinput, vl);
    vse16_v_f16m2(result, vacc, vl);
    
    printf("FMACC with LMUL=2, vl=16:\n");
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        float expected = 2.0f * (i + 1);
        float actual = (float)result[i];
        if (fabsf(actual - expected) > 0.1f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, expected, actual);
            errors++;
        }
    }
    printf("Result: %s\n", errors ? "FAILED" : "PASSED");
    
    // Test LMUL=1, vl=8
    vl = vsetvl_e16m1(8);
    vfloat16m1_t vinput1 = vle16_v_f16m1(input, vl);
    vfloat16m1_t vacc1 = vfmv_v_f_f16m1((__fp16)1.0, vl);  // Start with bias of 1.0
    
    // Perform FMACC: acc = 1.0 + 3.0 * input
    vacc1 = vfmacc_vf_f16m1(vacc1, (__fp16)3.0, vinput1, vl);
    vse16_v_f16m1(result, vacc1, vl);
    
    printf("\nFMACC with LMUL=1, vl=8, bias=1.0:\n");
    errors = 0;
    for (int i = 0; i < 8; i++) {
        float expected = 1.0f + 3.0f * (i + 1);
        float actual = (float)result[i];
        if (fabsf(actual - expected) > 0.1f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, expected, actual);
            errors++;
        }
    }
    printf("Result: %s\n", errors ? "FAILED" : "PASSED");
}

// Test 3: Strided memory access patterns (simulating reordered buffer access)
void test_strided_access() {
    printf("\n=== Test 3: Strided memory access patterns ===\n");
    
    // Simulate reordered buffer with different block sizes
    __fp16 buffer[128];
    __fp16 output[32];
    
    // Fill buffer with pattern
    for (int i = 0; i < 128; i++) {
        buffer[i] = (__fp16)(i * 0.1f);
    }
    
    // Test 1: Access 16-element block (simulating 16-column block in GEMM)
    printf("Test 16-element block access:\n");
    int vl = vsetvl_e16m2(16);
    __fp16 *ptr = buffer;
    vfloat16m2_t v = vle16_v_f16m2(ptr, vl);
    vse16_v_f16m2(output, v, vl);
    
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        if (fabsf((float)output[i] - (float)buffer[i]) > 0.001f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, (float)buffer[i], (float)output[i]);
            errors++;
        }
    }
    printf("16-element block: %s\n", errors ? "FAILED" : "PASSED");
    
    // Test 2: Access 8-element block with offset
    printf("\nTest 8-element block access with offset:\n");
    vl = vsetvl_e16m1(8);
    ptr = buffer + 16;  // Start after first 16-element block
    vfloat16m1_t v1 = vle16_v_f16m1(ptr, vl);
    vse16_v_f16m1(output, v1, vl);
    
    errors = 0;
    for (int i = 0; i < 8; i++) {
        if (fabsf((float)output[i] - (float)buffer[16 + i]) > 0.001f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, (float)buffer[16 + i], (float)output[i]);
            errors++;
        }
    }
    printf("8-element block with offset: %s\n", errors ? "FAILED" : "PASSED");
    
    // Test 3: Multiple small accesses (4, 2, 1)
    printf("\nTest small block accesses (4+2+1):\n");
    memset(output, 0, sizeof(output));
    
    // 4-element access
    vl = vsetvl_e16m1(4);
    ptr = buffer + 24;
    v1 = vle16_v_f16m1(ptr, vl);
    vse16_v_f16m1(output, v1, vl);
    
    // 2-element access
    vl = vsetvl_e16m1(2);
    ptr = buffer + 28;
    v1 = vle16_v_f16m1(ptr, vl);
    vse16_v_f16m1(output + 4, v1, vl);
    
    // 1-element access (scalar)
    output[6] = buffer[30];
    
    errors = 0;
    for (int i = 0; i < 7; i++) {
        if (fabsf((float)output[i] - (float)buffer[24 + i]) > 0.001f) {
            printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, (float)buffer[24 + i], (float)output[i]);
            errors++;
        }
    }
    printf("Combined small blocks: %s\n", errors ? "FAILED" : "PASSED");
}

// Test 4: Simulate GEMM tail computation pattern
void test_gemm_tail_pattern() {
    printf("\n=== Test 4: GEMM tail pattern simulation ===\n");
    
    // Simulate m=4, n=7, k=3 GEMM with tail
    const int m = 4, n = 7, k = 3;
    __fp16 kernel[12];  // m*k
    __fp16 input[21];   // k*n  
    __fp16 output[28];  // m*n
    __fp16 bias[4];
    
    // Initialize data
    for (int i = 0; i < m*k; i++) kernel[i] = (__fp16)(0.1f * (i + 1));
    for (int i = 0; i < k*n; i++) input[i] = (__fp16)(0.2f * (i + 1));
    for (int i = 0; i < m; i++) bias[i] = (__fp16)(0.5f * i);
    memset(output, 0, sizeof(output));
    
    // Compute reference result
    __fp16 ref_output[28];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ref_output[i*n + j] = bias[i];
            for (int c = 0; c < k; c++) {
                ref_output[i*n + j] += kernel[i*k + c] * input[c*n + j];
            }
        }
    }
    
    // Test vectorized computation with different vl values
    printf("Testing vectorized GEMM computation:\n");
    
    // Process first 4 columns (vl=4)
    int vl = vsetvl_e16m1(4);
    for (int i = 0; i < m; i++) {
        vfloat16m1_t vacc = vfmv_v_f_f16m1(bias[i], vl);
        for (int c = 0; c < k; c++) {
            vfloat16m1_t vin = vle16_v_f16m1(input + c*n, vl);
            vacc = vfmacc_vf_f16m1(vacc, kernel[i*k + c], vin, vl);
        }
        vse16_v_f16m1(output + i*n, vacc, vl);
    }
    
    // Process next 2 columns (vl=2)
    vl = vsetvl_e16m1(2);
    for (int i = 0; i < m; i++) {
        vfloat16m1_t vacc = vfmv_v_f_f16m1(bias[i], vl);
        for (int c = 0; c < k; c++) {
            vfloat16m1_t vin = vle16_v_f16m1(input + c*n + 4, vl);
            vacc = vfmacc_vf_f16m1(vacc, kernel[i*k + c], vin, vl);
        }
        vse16_v_f16m1(output + i*n + 4, vacc, vl);
    }
    
    // Process last column (scalar)
    for (int i = 0; i < m; i++) {
        __fp16 acc = bias[i];
        for (int c = 0; c < k; c++) {
            acc += kernel[i*k + c] * input[c*n + 6];
        }
        output[i*n + 6] = acc;
    }
    
    // Check results
    int errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < m*n; i++) {
        float diff = fabsf((float)output[i] - (float)ref_output[i]);
        if (diff > 0.01f) {
            if (errors < 5) {  // Print first 5 errors
                printf("  ERROR at [%d,%d]: expected %.4f, got %.4f, diff=%.4f\n", 
                       i/n, i%n, (float)ref_output[i], (float)output[i], diff);
            }
            errors++;
        }
        if (diff > max_error) max_error = diff;
    }
    printf("Errors: %d/%d, Max error: %.6f\n", errors, m*n, max_error);
    printf("Result: %s\n", errors ? "FAILED" : "PASSED");
}

// Test 5: Edge case - very small vl values
void test_small_vl() {
    printf("\n=== Test 5: Small VL edge cases ===\n");
    
    __fp16 src[16], dst[16];
    for (int i = 0; i < 16; i++) {
        src[i] = (__fp16)(i * 0.25f);
        dst[i] = 0;
    }
    
    // Test vl=1
    int vl = vsetvl_e16m1(1);
    printf("vsetvl_e16m1(1) returned vl=%d\n", vl);
    if (vl == 1) {
        vfloat16m1_t v = vle16_v_f16m1(src, vl);
        vse16_v_f16m1(dst, v, vl);
        
        if (fabsf((float)dst[0] - (float)src[0]) > 0.001f) {
            printf("  ERROR: vl=1 failed, expected %.2f, got %.2f\n", (float)src[0], (float)dst[0]);
        } else {
            printf("  vl=1: PASSED\n");
        }
    } else {
        printf("  WARNING: vl=1 not supported on this hardware\n");
    }
    
    // Test vl=2
    memset(dst, 0, sizeof(dst));
    vl = vsetvl_e16m1(2);
    printf("\nvsetvl_e16m1(2) returned vl=%d\n", vl);
    if (vl >= 2) {
        vfloat16m1_t v = vle16_v_f16m1(src, vl);
        vse16_v_f16m1(dst, v, vl);
        
        int errors = 0;
        for (int i = 0; i < 2; i++) {
            if (fabsf((float)dst[i] - (float)src[i]) > 0.001f) {
                printf("  ERROR at [%d]: expected %.2f, got %.2f\n", i, (float)src[i], (float)dst[i]);
                errors++;
            }
        }
        printf("  vl=2: %s\n", errors ? "FAILED" : "PASSED");
    }
}

// Test 6: Check for off-by-one errors in pointer arithmetic
void test_pointer_arithmetic() {
    printf("\n=== Test 6: Pointer arithmetic patterns ===\n");
    
    __fp16 buffer[64];
    for (int i = 0; i < 64; i++) {
        buffer[i] = (__fp16)(i);
    }
    
    // Simulate accessing blocks at calculated offsets
    size_t offsets[] = {0, 16, 24, 28, 30, 31};  // 16 + 8 + 4 + 2 + 1 pattern
    int block_sizes[] = {16, 8, 4, 2, 1, 1};
    
    printf("Testing block access at calculated offsets:\n");
    for (int i = 0; i < 6; i++) {
        __fp16 result[16];
        memset(result, 0, sizeof(result));
        
        if (block_sizes[i] >= 16) {
            int vl = vsetvl_e16m2(block_sizes[i]);
            vfloat16m2_t v = vle16_v_f16m2(buffer + offsets[i], vl);
            vse16_v_f16m2(result, v, vl);
        } else if (block_sizes[i] >= 2) {
            int vl = vsetvl_e16m1(block_sizes[i]);
            vfloat16m1_t v = vle16_v_f16m1(buffer + offsets[i], vl);
            vse16_v_f16m1(result, v, vl);
        } else {
            result[0] = buffer[offsets[i]];
        }
        
        int errors = 0;
        for (int j = 0; j < block_sizes[i]; j++) {
            if (fabsf((float)result[j] - (float)buffer[offsets[i] + j]) > 0.001f) {
                printf("  ERROR: Block %d, element %d: expected %.0f, got %.0f\n", 
                       i, j, (float)buffer[offsets[i] + j], (float)result[j]);
                errors++;
            }
        }
        printf("Block %d (offset=%zu, size=%d): %s\n", 
               i, offsets[i], block_sizes[i], errors ? "FAILED" : "PASSED");
    }
}

int main() {
    printf("RVV Tail Processing Pattern Test\n");
    printf("=================================\n");
    
    // Get VLEN
    int vl = vsetvl_e8m1(256);
    printf("VLEN = %d bits\n", vl * 8);
    
    test_basic_vle_vse();
    test_vfmacc_patterns();
    test_strided_access();
    test_gemm_tail_pattern();
    test_small_vl();
    test_pointer_arithmetic();
    
    printf("\n=================================\n");
    printf("Test completed.\n");
    
    return 0;
}
