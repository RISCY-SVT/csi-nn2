/*
 * Minimal conv2d test with simple known values
 * Tests 1x1x3x3 input, 1x1x3x3 kernel -> 1x1x3x3 output
 */

#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>
#include <math.h>

static void test_minimal_conv2d(void)
{
    printf("\n=== Minimal Conv2D Test ===\n");
    
    // Simple 3x3 input with values 1-9
    float input_data[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    
    // Simple 3x3 kernel (identity-like)
    float kernel_data[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    
    // Zero bias
    float bias_data[1] = {0.0f};
    
    // Expected output for 3x3 conv with padding=1, stride=1
    // Should be same as input for this kernel
    float expected_output[9] = {
        5.0f, 0.0f, 0.0f,  // Due to padding
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    
    // Setup tensors
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;  // batch
    input->dim[1] = 1;  // channels
    input->dim[2] = 3;  // height
    input->dim[3] = 3;  // width
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->data = input_data;
    
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = 1;  // out_channels
    kernel->dim[1] = 1;  // in_channels
    kernel->dim[2] = 3;  // kernel_height
    kernel->dim[3] = 3;  // kernel_width
    kernel->dim_count = 4;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    kernel->layout = CSINN_LAYOUT_OIHW;
    kernel->data = kernel_data;
    
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = 1;  // out_channels
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    bias->layout = CSINN_LAYOUT_O;
    bias->data = bias_data;
    
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;  // batch
    output->dim[1] = 1;  // out_channels
    output->dim[2] = 3;  // out_height
    output->dim[3] = 3;  // out_width
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    
    float output_data[9];
    memset(output_data, 0, sizeof(output_data));
    output->data = output_data;
    
    struct csinn_conv2d_params *params = 
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    
    // First, test kernel reordering
    printf("\nTesting kernel reorder...\n");
    float kernel_copy[9];
    memcpy(kernel_copy, kernel_data, sizeof(kernel_copy));
    kernel->data = kernel_copy;
    
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    printf("Original kernel:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", kernel_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    printf("\nReordered kernel:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", kernel_copy[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Now test computation with reordered kernel
    printf("\nTesting computation...\n");
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("Compute returned: %d\n", ret);
    
    printf("\nInput:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", input_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    printf("\nOutput:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", output_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Check for errors
    int errors = 0;
    for (int i = 0; i < 9; i++) {
        if (fabs(output_data[i] - expected_output[i]) > 1e-5) {
            printf("ERROR at [%d]: got %.3f, expected %.3f\n", 
                   i, output_data[i], expected_output[i]);
            errors++;
        }
    }
    
    printf("\nTest %s: %d errors\n", errors == 0 ? "PASSED" : "FAILED", errors);
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
}

static void test_conv2d_no_padding(void)
{
    printf("\n=== Conv2D Test Without Padding ===\n");
    
    // 5x5 input
    float input_data[25] = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9,  10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    
    // 3x3 averaging kernel
    float kernel_data[9] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    
    float bias_data[1] = {0.0f};
    
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
    output->dim[2] = 3;  // (5-3)/1 + 1 = 3
    output->dim[3] = 3;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
    
    float output_data[9];
    memset(output_data, 0, sizeof(output_data));
    output->data = output_data;
    
    struct csinn_conv2d_params *params = 
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 0;
    params->pad_right = 0;
    params->pad_top = 0;
    params->pad_down = 0;
    params->group = 1;
    
    // Reorder kernel
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    // Compute
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("Compute returned: %d\n", ret);
    
    printf("\nOutput (should be averages of 3x3 regions):\n");
    for (int i = 0; i < 9; i++) {
        printf("%.2f ", output_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Expected: center of each 3x3 region
    float expected[9] = {
        7.0f, 8.0f, 9.0f,
        12.0f, 13.0f, 14.0f,
        17.0f, 18.0f, 19.0f
    };
    
    int errors = 0;
    for (int i = 0; i < 9; i++) {
        if (fabs(output_data[i] - expected[i]) > 0.1) {
            printf("ERROR at [%d]: got %.3f, expected %.3f\n", 
                   i, output_data[i], expected[i]);
            errors++;
        }
    }
    
    printf("\nTest %s: %d errors\n", errors == 0 ? "PASSED" : "FAILED", errors);
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
}

int main(int argc, char **argv)
{
    init_testsuite("Minimal Conv2D im2col_gemm tests for RVV.\n");
    
    test_minimal_conv2d();
    test_conv2d_no_padding();
    
    return done_testing();
}
