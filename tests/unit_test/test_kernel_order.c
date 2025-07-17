/*
 * Test if kernel ordering is the issue
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>

static void test_with_original_kernel()
{
    printf("=== Testing with original (non-reordered) kernel ===\n");
    
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = 3;
    input->dim[2] = 4;
    input->dim[3] = 5;
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = 19;
    kernel->dim[1] = 3;
    kernel->dim[2] = 3;
    kernel->dim[3] = 3;
    kernel->dim_count = 4;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    kernel->layout = CSINN_LAYOUT_OIHW;
    
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = 19;
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    bias->layout = CSINN_LAYOUT_O;
    
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = 19;
    output->dim[2] = 4;
    output->dim[3] = 5;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    
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
    params->dilation_height = 1;
    params->dilation_width = 1;
    
    // Test 1: With already reordered kernel (conv2d_im2col_fp32_ker1)
    printf("\nTest 1: Using pre-reordered kernel (ker1):\n");
    input->data = conv2d_im2col_fp32_in;
    kernel->data = conv2d_im2col_fp32_ker1; // Already reordered!
    bias->data = conv2d_im2col_fp32_bias;
    
    float *output1 = shl_mem_alloc(380 * sizeof(float));
    output->data = output1;
    
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("  RVV returned: %d\n", ret);
    printf("  First 5 outputs: ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", output1[i]);
    }
    printf("\n");
    
    // Test 2: With original kernel that needs reordering
    printf("\nTest 2: Using original kernel (ker) with reorder:\n");
    
    // Copy original kernel
    float *kernel_copy = shl_mem_alloc(513 * sizeof(float));
    memcpy(kernel_copy, conv2d_im2col_fp32_ker, 513 * sizeof(float));
    kernel->data = kernel_copy;
    
    // Reorder it
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    float *output2 = shl_mem_alloc(380 * sizeof(float));
    output->data = output2;
    
    ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("  RVV returned: %d\n", ret);
    printf("  First 5 outputs: ");
    for (int i = 0; i < 5; i++) {
        printf("%.3f ", output2[i]);
    }
    printf("\n");
    
    // Compare outputs
    printf("\nComparing outputs:\n");
    int diffs = 0;
    for (int i = 0; i < 380 && diffs < 10; i++) {
        float diff = fabsf(output1[i] - output2[i]);
        if (diff > 0.001) {
            printf("  Diff at [%d]: method1=%.3f, method2=%.3f, diff=%.3f\n",
                   i, output1[i], output2[i], diff);
            diffs++;
        }
    }
    if (diffs == 0) {
        printf("  Outputs are identical\n");
    }
    
    // Test 3: Initialize through csinn_conv2d_init
    printf("\nTest 3: Through proper initialization:\n");
    kernel->data = conv2d_im2col_fp32_ker; // Original kernel
    
    ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  Init returned: %d\n", ret);
    
    if (ret == CSINN_TRUE && params->base.cb && params->base.cb->exec) {
        float *output3 = shl_mem_alloc(380 * sizeof(float));
        output->data = output3;
        
        ret = csinn_conv2d(input, output, kernel, bias, params);
        printf("  Conv2d returned: %d\n", ret);
        printf("  First 5 outputs: ");
        for (int i = 0; i < 5; i++) {
            printf("%.3f ", output3[i]);
        }
        printf("\n");
        
        shl_mem_free(output3);
    }
    
    // Evaluate against reference
    printf("\nEvaluating method 2 against reference:\n");
    evaluate_error(output2, conv2d_im2col_fp32_out, 380, CSINN_DTYPE_FLOAT32);
    
    shl_mem_free(output1);
    shl_mem_free(output2);
    shl_mem_free(kernel_copy);
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

int main(int argc, char **argv)
{
    init_testsuite("Testing kernel ordering issues.\n");
    
    test_with_original_kernel();
    
    return done_testing();
}
