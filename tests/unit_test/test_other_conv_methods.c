/*
 * Test other convolution methods
 */

#include "./valid_data/conv2d.dat"
#include "rvv/rvv.h"
#include "csi_nn.h"
#include "test_utils.h"

static void test_simple_conv_with_different_methods()
{
    printf("=== Testing different convolution methods ===\n\n");
    
    // Simple test data
    float input_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel_data[9] = {1, 0, 0, 0, 0, 0, 0, 0, 0}; // Top-left = 1
    float bias_data[1] = {10.0};
    
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
    
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    params->dilation_height = 1;
    params->dilation_width = 1;
    
    printf("Expected output (correct convolution):\n");
    printf("10.0 10.0 10.0\n");
    printf("10.0 11.0 12.0\n");
    printf("10.0 14.0 15.0\n\n");
    
    // Test 1: Through proper init
    printf("Test 1: Through csinn_conv2d_init/csinn_conv2d:\n");
    float output1[9] = {0};
    output->data = output1;
    
    params->base.api = CSINN_C920;
    int ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  Init returned: %d\n", ret);
    
    if (ret == CSINN_TRUE && params->base.cb && params->base.cb->exec) {
        ret = csinn_conv2d(input, output, kernel, bias, params);
        printf("  Conv2d returned: %d\n", ret);
        printf("  Output:\n");
        for (int i = 0; i < 9; i++) {
            printf("  %.1f", output1[i]);
            if ((i + 1) % 3 == 0) printf("\n");
        }
    } else {
        printf("  No callback assigned\n");
    }
    
    // Test 2: Try reference implementation
    printf("\nTest 2: Reference implementation (CSINN_REF):\n");
    float output2[9] = {0};
    output->data = output2;
    
    params->base.api = CSINN_REF;
    params->base.cb = NULL;
    ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  Init returned: %d\n", ret);
    
    if (ret == CSINN_TRUE && params->base.cb && params->base.cb->exec) {
        ret = csinn_conv2d(input, output, kernel, bias, params);
        printf("  Conv2d returned: %d\n", ret);
        printf("  Output:\n");
        for (int i = 0; i < 9; i++) {
            printf("  %.1f", output2[i]);
            if ((i + 1) % 3 == 0) printf("\n");
        }
    } else {
        printf("  No callback assigned\n");
    }
    
    // Test 3: Try direct 3x3 convolution (fp16 NHWC version exists)
    printf("\nTest 3: Direct 3x3 convolution:\n");
    printf("  Direct 3x3 convolution only available for fp16 NHWC layout\n");
    printf("  Skipping this test\n");
    
    // Test 4: Try im2col gemm directly
    printf("\nTest 4: Direct im2col gemm:\n");
    float output4[9] = {0};
    output->data = output4;
    params->base.api = CSINN_C920;
    
    ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("  shl_rvv_conv_im2col_gemm_fp32 returned: %d\n", ret);
    printf("  Output:\n");
    for (int i = 0; i < 9; i++) {
        printf("  %.1f", output4[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Test 5: Test with original test data
    printf("\nTest 5: With original test data (first 3x3 of input):\n");
    float test_input[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            test_input[i*3 + j] = conv2d_im2col_fp32_in[i*5 + j]; // First 3x3 of 3x4x5
        }
    }
    
    float test_kernel[9];
    for (int i = 0; i < 9; i++) {
        test_kernel[i] = conv2d_im2col_fp32_ker[i]; // First kernel
    }
    
    float test_bias[1] = {conv2d_im2col_fp32_bias[0]};
    float output5[9] = {0};
    
    input->data = test_input;
    kernel->data = test_kernel;
    bias->data = test_bias;
    output->data = output5;
    
    // Reorder kernel for im2col_gemm
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("  Return: %d\n", ret);
    printf("  Output:\n");
    for (int i = 0; i < 9; i++) {
        printf("  %.3f", output5[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

int main(int argc, char **argv)
{
    init_testsuite("Testing other convolution methods.\n");
    
    test_simple_conv_with_different_methods();
    
    return done_testing();
}
