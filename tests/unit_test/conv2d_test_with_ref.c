/*
 * Test conv2d with proper reference implementation
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>

static void test_conv2d_implementations(void)
{
    printf("\n=== Testing different Conv2D implementations ===\n");
    
    // Setup simple test data
    float input_data[60];
    float kernel_data[513];
    float bias_data[19];
    float output_rvv[380];
    float output_ref[380];
    
    // Initialize with simple pattern
    for (int i = 0; i < 60; i++) {
        input_data[i] = i * 0.1f;
    }
    
    for (int i = 0; i < 513; i++) {
        kernel_data[i] = (i % 9) * 0.01f;
    }
    
    for (int i = 0; i < 19; i++) {
        bias_data[i] = i * 0.5f;
    }
    
    // Setup tensors
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = 3;
    input->dim[2] = 4;
    input->dim[3] = 5;
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    input->data = input_data;
    
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
    bias->data = bias_data;
    
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
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    params->base.api = CSINN_REF;  // Try different APIs
    
    // Test 1: RVV implementation
    printf("\n--- Testing RVV implementation ---\n");
    memcpy(output_rvv, bias_data, 19 * sizeof(float)); // Pre-fill with bias
    memset(output_rvv + 19, 0, (380 - 19) * sizeof(float));
    output->data = output_rvv;
    
    // Copy and reorder kernel for RVV
    float kernel_rvv[513];
    memcpy(kernel_rvv, kernel_data, 513 * sizeof(float));
    kernel->data = kernel_rvv;
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    int ret_rvv = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("RVV returned: %d\n", ret_rvv);
    printf("First 10 outputs: ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", output_rvv[i]);
    }
    printf("\n");
    
    // Test 2: Reference implementation
    printf("\n--- Testing reference implementation ---\n");
    memset(output_ref, 0, 380 * sizeof(float));
    output->data = output_ref;
    kernel->data = kernel_data; // Use original kernel
    
    // Try different reference functions
    printf("Trying shl_ref_conv2d_f32...\n");
    int ret_ref = shl_ref_conv2d_f32(input, output, kernel, bias, params);
    printf("shl_ref_conv2d_f32 returned: %d\n", ret_ref);
    
    if (ret_ref != 0) {
        // If that fails, try the C reference
        printf("\nTrying csi_ref_conv2d_f32...\n");
        params->base.api = CSINN_C920;  // Try different API
        ret_ref = csi_ref_conv2d_f32(input, output, kernel, bias, params);
        printf("csi_ref_conv2d_f32 returned: %d\n", ret_ref);
    }
    
    printf("First 10 outputs: ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", output_ref[i]);
    }
    printf("\n");
    
    // Compare
    if (ret_ref == 0) {
        printf("\n--- Comparing RVV vs Reference ---\n");
        int diffs = 0;
        for (int i = 0; i < 380 && diffs < 10; i++) {
            float diff = fabsf(output_rvv[i] - output_ref[i]);
            if (diff > 0.001) {
                printf("Diff at [%d]: RVV=%.3f, REF=%.3f, diff=%.3f\n",
                       i, output_rvv[i], output_ref[i], diff);
                diffs++;
            }
        }
    }
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
}

static void test_simple_3x3(void)
{
    printf("\n=== Simple 3x3 Conv Test (no padding, stride=1) ===\n");
    
    // 5x5 input
    float input_data[25] = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9, 10,
        11,12, 13, 14, 15,
        16,17, 18, 19, 20,
        21,22, 23, 24, 25
    };
    
    // 3x3 kernel (simple sum)
    float kernel_data[9] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    
    float bias_data[1] = {0.0f};
    float output_data[9] = {0};
    
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
    output->dim[2] = 3;
    output->dim[3] = 3;
    output->dim_count = 4;
    output->dtype = CSINN_DTYPE_FLOAT32;
    output->layout = CSINN_LAYOUT_NCHW;
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
    params->base.api = CSINN_REF;
    
    // Test with RVV
    printf("Testing RVV (after reorder)...\n");
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    
    printf("Return code: %d\n");
    printf("Output:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", output_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Expected: sum of each 3x3 window
    float expected[9] = {
        63,  72,  81,   // sums of top 3x3 windows
        108, 117, 126,  // middle row
        153, 162, 171   // bottom row
    };
    
    printf("\nExpected:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", expected[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
}

int main(int argc, char **argv)
{
    init_testsuite("Testing Conv2D implementations\n");
    
    test_simple_3x3();
    test_conv2d_implementations();
    
    return done_testing();
}
