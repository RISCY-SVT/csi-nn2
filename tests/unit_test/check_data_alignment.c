/*
 * Check data alignment and sizes for RVV
 */

#include "./valid_data/conv2d.dat"
#include <stdio.h>
#include <stdint.h>

#define CHECK_ALIGNMENT(ptr, name, align) do { \
    uintptr_t addr = (uintptr_t)(ptr); \
    printf("%-30s: addr=%p, aligned to %d bytes: %s\n", \
           name, ptr, align, (addr % align == 0) ? "YES" : "NO"); \
} while(0)

int main() {
    printf("=== Checking data alignment for RVV ===\n\n");
    
    // Check FP32 data
    printf("--- FP32 data ---\n");
    CHECK_ALIGNMENT(conv2d_im2col_fp32_in, "conv2d_im2col_fp32_in", 4);
    CHECK_ALIGNMENT(conv2d_im2col_fp32_in, "conv2d_im2col_fp32_in", 8);
    CHECK_ALIGNMENT(conv2d_im2col_fp32_in, "conv2d_im2col_fp32_in", 16);
    
    CHECK_ALIGNMENT(conv2d_im2col_fp32_ker, "conv2d_im2col_fp32_ker", 4);
    CHECK_ALIGNMENT(conv2d_im2col_fp32_ker1, "conv2d_im2col_fp32_ker1", 4);
    CHECK_ALIGNMENT(conv2d_im2col_fp32_bias, "conv2d_im2col_fp32_bias", 4);
    CHECK_ALIGNMENT(conv2d_im2col_fp32_out, "conv2d_im2col_fp32_out", 4);
    
    printf("\n--- FP16 data ---\n");
    CHECK_ALIGNMENT(conv2d_im2col_fp16_in, "conv2d_im2col_fp16_in", 2);
    CHECK_ALIGNMENT(conv2d_im2col_fp16_in, "conv2d_im2col_fp16_in", 4);
    CHECK_ALIGNMENT(conv2d_im2col_fp16_in, "conv2d_im2col_fp16_in", 8);
    
    // Check sizes (based on test parameters)
    printf("\n=== Expected sizes ===\n");
    printf("Input:  1x3x4x5 = %d elements\n", 1*3*4*5);
    printf("Kernel: 19x3x3x3 = %d elements\n", 19*3*3*3);
    printf("Bias:   19 elements\n");
    printf("Output: 1x19x4x5 = %d elements\n", 1*19*4*5);
    
    // Dump first few values to check if data is valid
    printf("\n=== First few values ===\n");
    printf("FP32 input[0..4]: ");
    for (int i = 0; i < 5 && i < 60; i++) {
        printf("%.3f ", conv2d_im2col_fp32_in[i]);
    }
    printf("\n");
    
    printf("FP32 kernel[0..4]: ");
    for (int i = 0; i < 5 && i < 513; i++) {
        printf("%.3f ", conv2d_im2col_fp32_ker[i]);
    }
    printf("\n");
    
    printf("FP32 bias[0..4]: ");
    for (int i = 0; i < 5 && i < 19; i++) {
        printf("%.3f ", conv2d_im2col_fp32_bias[i]);
    }
    printf("\n");
    
    printf("FP32 expected_out[0..4]: ");
    for (int i = 0; i < 5 && i < 380; i++) {
        printf("%.3f ", conv2d_im2col_fp32_out[i]);
    }
    printf("\n");
    
    return 0;
}
