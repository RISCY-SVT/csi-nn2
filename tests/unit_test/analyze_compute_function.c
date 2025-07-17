/*
 * Analyze what the compute function actually does
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>

static void test_identity_convolution()
{
    printf("=== Testing identity convolution ===\n");
    
    // 1x1x5x5 input
    float input_data[25] = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    
    // 1x1x3x3 identity kernel (center = 1, others = 0)
    float kernel_data[9] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    
    float bias_data[1] = {100.0}; // Large bias to make it obvious
    float output_data[25] = {0};
    
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = 1;
    input->dim[2] = 5;
    input->dim[3] = 5;
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
    output->dim[2] = 5;
    output->dim[3] = 5;
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
    
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    
    printf("Return: %d\n\n", ret);
    printf("Output:\n");
    for (int i = 0; i < 25; i++) {
        printf("%6.1f ", output_data[i]);
        if ((i + 1) % 5 == 0) printf("\n");
    }
    
    printf("\nExpected (input + bias at valid positions):\n");
    for (int h = 0; h < 5; h++) {
        for (int w = 0; w < 5; w++) {
            printf("%6.1f ", input_data[h * 5 + w] + 100.0);
        }
        printf("\n");
    }
    
    // Check if it's just adding bias
    printf("\nChecking if output = input + bias:\n");
    int matches = 0;
    for (int i = 0; i < 25; i++) {
        if (fabsf(output_data[i] - (input_data[i] + bias_data[0])) < 0.001) {
            matches++;
        }
    }
    printf("Matches: %d/25\n", matches);
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

static void test_zero_kernel()
{
    printf("\n=== Testing with zero kernel ===\n");
    
    float input_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel_data[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; // All zeros
    float bias_data[1] = {42.0};
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
    
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    
    printf("With zero kernel, output should be all bias (42.0):\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", output_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

static void test_without_padding()
{
    printf("\n=== Testing without padding ===\n");
    
    float input_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel_data[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1}; // All ones (sum filter)
    float bias_data[1] = {0.0};
    float output_data[1] = {0}; // Output will be 1x1
    
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
    output->dim[2] = 1;
    output->dim[3] = 1;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    output->data = output_data;
    
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->base.api = CSINN_C920;
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 0;  // No padding
    params->pad_right = 0;
    params->pad_top = 0;
    params->pad_down = 0;
    params->group = 1;
    
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    
    printf("Without padding, 3x3 kernel on 3x3 input:\n");
    printf("Output: %.1f\n", output_data[0]);
    printf("Expected: %.1f (sum of all input elements)\n", 1+2+3+4+5+6+7+8+9.0);
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

int main(int argc, char **argv)
{
    init_testsuite("Analyze what compute function actually does.\n");
    
    test_identity_convolution();
    test_zero_kernel();
    test_without_padding();
    
    return done_testing();
}
