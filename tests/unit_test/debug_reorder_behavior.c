/*
 * Debug kernel reorder behavior
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>

static void print_kernel_slice(float *kernel, int out_ch, int in_ch, int k_h, int k_w, const char *label)
{
    printf("\n%s (first output channel, first input channel):\n", label);
    for (int h = 0; h < k_h; h++) {
        for (int w = 0; w < k_w; w++) {
            // OIHW layout: kernel[oc][ic][h][w]
            int idx = ((0 * in_ch + 0) * k_h + h) * k_w + w;
            printf("%8.3f ", kernel[idx]);
        }
        printf("\n");
    }
}

static void test_reorder_behavior()
{
    printf("=== Testing kernel reorder behavior ===\n");
    
    // Test dimensions
    int out_c = 19, in_c = 3, k_h = 3, k_w = 3;
    int kernel_size = out_c * in_c * k_h * k_w;
    
    // Create copies of kernel data
    float *kernel_orig = shl_mem_alloc(kernel_size * sizeof(float));
    float *kernel_for_reorder = shl_mem_alloc(kernel_size * sizeof(float));
    float *kernel_pre_reordered = shl_mem_alloc(kernel_size * sizeof(float));
    
    memcpy(kernel_orig, conv2d_im2col_fp32_ker, kernel_size * sizeof(float));
    memcpy(kernel_for_reorder, conv2d_im2col_fp32_ker, kernel_size * sizeof(float));
    memcpy(kernel_pre_reordered, conv2d_im2col_fp32_ker1, kernel_size * sizeof(float));
    
    // Setup tensor and params for reorder
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = out_c;
    kernel->dim[1] = in_c;
    kernel->dim[2] = k_h;
    kernel->dim[3] = k_w;
    kernel->dim_count = 4;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    kernel->layout = CSINN_LAYOUT_OIHW;
    
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->base.api = CSINN_C920;
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    
    // Print original kernel slice
    print_kernel_slice(kernel_orig, out_c, in_c, k_h, k_w, "Original kernel (ker)");
    
    // Apply reorder
    kernel->data = kernel_for_reorder;
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    // Print reordered kernel slice
    print_kernel_slice(kernel_for_reorder, out_c, in_c, k_h, k_w, "After reorder");
    
    // Print pre-reordered kernel slice
    print_kernel_slice(kernel_pre_reordered, out_c, in_c, k_h, k_w, "Pre-reordered (ker1)");
    
    // Compare reordered with pre-reordered
    printf("\nComparing reordered with pre-reordered:\n");
    int diffs = 0;
    for (int i = 0; i < kernel_size && diffs < 10; i++) {
        float diff = fabsf(kernel_for_reorder[i] - kernel_pre_reordered[i]);
        if (diff > 0.001) {
            printf("  Diff at [%d]: reordered=%.3f, pre-reordered=%.3f\n",
                   i, kernel_for_reorder[i], kernel_pre_reordered[i]);
            diffs++;
        }
    }
    if (diffs == 0) {
        printf("  Reordered kernel matches pre-reordered kernel (ker1)\n");
    }
    
    // Now test if function modifies kernel internally
    printf("\n=== Testing if compute function modifies kernel ===\n");
    
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = 4;
    input->dim[3] = 5;
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->data = conv2d_im2col_fp32_in;
    
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = out_c;
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    bias->layout = CSINN_LAYOUT_O;
    bias->data = conv2d_im2col_fp32_bias;
    
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_c;
    output->dim[2] = 4;
    output->dim[3] = 5;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    
    // Test 1: With original kernel
    float *kernel_test1 = shl_mem_alloc(kernel_size * sizeof(float));
    memcpy(kernel_test1, conv2d_im2col_fp32_ker, kernel_size * sizeof(float));
    kernel->data = kernel_test1;
    
    float *output1 = shl_mem_alloc(380 * sizeof(float));
    output->data = output1;
    
    // Save kernel checksum before call
    float checksum_before = 0;
    for (int i = 0; i < kernel_size; i++) {
        checksum_before += kernel_test1[i];
    }
    
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    
    // Check kernel checksum after call
    float checksum_after = 0;
    for (int i = 0; i < kernel_size; i++) {
        checksum_after += kernel_test1[i];
    }
    
    printf("Test 1 (original kernel):\n");
    printf("  Return: %d\n", ret);
    printf("  Kernel checksum before: %.3f\n", checksum_before);
    printf("  Kernel checksum after: %.3f\n", checksum_after);
    printf("  Kernel modified: %s\n", 
           fabsf(checksum_before - checksum_after) > 0.001 ? "YES" : "NO");
    printf("  First 5 outputs: ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", output1[i]);
    }
    printf("\n");
    
    // Test 2: With pre-reordered kernel
    float *kernel_test2 = shl_mem_alloc(kernel_size * sizeof(float));
    memcpy(kernel_test2, conv2d_im2col_fp32_ker1, kernel_size * sizeof(float));
    kernel->data = kernel_test2;
    
    float *output2 = shl_mem_alloc(380 * sizeof(float));
    output->data = output2;
    
    ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    
    printf("\nTest 2 (pre-reordered kernel):\n");
    printf("  Return: %d\n", ret);
    printf("  First 5 outputs: ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", output2[i]);
    }
    printf("\n");
    
    // Clean up
    shl_mem_free(kernel_orig);
    shl_mem_free(kernel_for_reorder);
    shl_mem_free(kernel_pre_reordered);
    shl_mem_free(kernel_test1);
    shl_mem_free(kernel_test2);
    shl_mem_free(output1);
    shl_mem_free(output2);
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

static void test_simple_case_detailed()
{
    printf("\n=== Testing simple case with detailed output ===\n");
    
    // Very simple 1x1x3x3 case
    float input_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel_data[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0}; // Top-left corner = 1
    float bias_data[1] = {10.0};
    float output_data[9] = {0};
    
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = 1;
    input->dim[2] = 3;
    input->dim[3] = 3;
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->data = input_data;
    
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = 1;
    kernel->dim[1] = 1;
    kernel->dim[2] = 3;
    kernel->dim[3] = 3;
    kernel->dim_count = 4;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    kernel->layout = CSINN_LAYOUT_OIHW;
    kernel->data = kernel_data;
    
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = 1;
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    bias->layout = CSINN_LAYOUT_O;
    bias->data = bias_data;
    
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = 1;
    output->dim[2] = 3;
    output->dim[3] = 3;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->data = output_data;
    
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->base.api = CSINN_C920;
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    
    printf("Input:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.0f ", input_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    printf("\nKernel:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.0f ", kernel_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    printf("\nBias: %.0f\n", bias_data[0]);
    
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    
    printf("\nOutput after compute:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", output_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    printf("\nExpected (with padding, stride=1):\n");
    printf("10.0 10.0 10.0\n");
    printf("10.0 11.0 12.0\n");  // 11 = 1*1 + 10 (bias)
    printf("10.0 14.0 15.0\n");
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

int main(int argc, char **argv)
{
    init_testsuite("Debug kernel reorder behavior.\n");
    
    test_reorder_behavior();
    test_simple_case_detailed();
    
    return done_testing();
}
