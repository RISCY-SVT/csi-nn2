/*
 * Test conv2d with correct C920 API
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>
#include <math.h>

static void test_conv2d_with_proper_api(void)
{
    printf("\n=== Testing Conv2D with C920 API ===\n");
    
    // Simple test data
    float input_data[60] = {0};
    float kernel_data[513] = {0};
    float bias_data[19] = {0};
    float output_data[380] = {0};
    
    // Initialize with simple patterns
    for (int i = 0; i < 60; i++) {
        input_data[i] = (i % 10) * 0.1f;
    }
    
    // Simple kernel - mostly zeros with some ones
    for (int i = 0; i < 513; i++) {
        kernel_data[i] = (i % 27 == 13) ? 1.0f : 0.0f;
    }
    
    for (int i = 0; i < 19; i++) {
        bias_data[i] = i * 0.1f;
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
    kernel->data = kernel_data;
    
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
    params->base.api = CSINN_C920;  // Use correct C920 API
    
    printf("Using API: CSINN_C920 (%d)\n", CSINN_C920);
    
    // Method 1: Direct RVV call (what the test was doing)
    printf("\n--- Method 1: Direct RVV call ---\n");
    memset(output_data, 0, 380 * sizeof(float));
    
    // Reorder kernel for im2col_gemm
    float kernel_reordered[513];
    memcpy(kernel_reordered, kernel_data, 513 * sizeof(float));
    kernel->data = kernel_reordered;
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    int ret1 = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("Direct RVV call returned: %d\n", ret1);
    printf("First 10 outputs: ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", output_data[i]);
    }
    printf("\n");
    
    // Method 2: Through initialization
    printf("\n--- Method 2: Through init/compute sequence ---\n");
    memset(output_data, 0, 380 * sizeof(float));
    kernel->data = kernel_data;  // Reset to original kernel
    
    // Initialize conv2d
    int ret_init = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("csinn_conv2d_init returned: %d\n", ret_init);
    
    if (ret_init == CSINN_TRUE && params->base.cb && params->base.cb->exec) {
        printf("Compute function was assigned: YES\n");
        
        // Allocate data if needed
        input->data = input_data;
        kernel->data = kernel_data;
        bias->data = bias_data;
        output->data = output_data;
        
        // Call compute
        int ret_compute = csinn_conv2d(input, output, kernel, bias, params);
        printf("csinn_conv2d returned: %d\n", ret_compute);
        
        printf("First 10 outputs: ");
        for (int i = 0; i < 10; i++) {
            printf("%.3f ", output_data[i]);
        }
        printf("\n");
    } else {
        printf("Compute function was NOT properly assigned\n");
        if (!params->base.cb) {
            printf("  params->base.cb is NULL\n");
        } else if (!params->base.cb->exec) {
            printf("  params->base.cb->exec is NULL\n");
        }
    }
    
    // Method 3: Use reference implementation for comparison
    printf("\n--- Method 3: Reference implementation ---\n");
    memset(output_data, 0, 380 * sizeof(float));
    kernel->data = kernel_data;  // Reset to original
    params->base.api = CSINN_REF;  // Switch to reference
    
    ret_init = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("Reference init returned: %d\n", ret_init);
    
    if (ret_init == CSINN_TRUE && params->base.cb && params->base.cb->exec) {
        int ret_ref = csinn_conv2d(input, output, kernel, bias, params);
        printf("Reference compute returned: %d\n", ret_ref);
        
        printf("First 10 outputs: ");
        for (int i = 0; i < 10; i++) {
            printf("%.3f ", output_data[i]);
        }
        printf("\n");
    } else {
        printf("Reference implementation not properly initialized\n");
    }
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
}

static void test_check_session_setup(void)
{
    printf("\n=== Testing Session Setup ===\n");
    
    // Create a simple session
    struct csinn_session *sess = csinn_alloc_session();
    sess->base_api = CSINN_C920;
    sess->base_dtype = CSINN_DTYPE_FLOAT32;
    sess->base_layout = CSINN_LAYOUT_NCHW;
    sess->base_run_mode = CSINN_RM_LAYER;
    sess->base_quant_type = CSINN_QUANT_FLOAT32;
    
    printf("Session API: %d (CSINN_C920=%d)\n", sess->base_api, CSINN_C920);
    printf("Base dtype: %d\n", sess->base_dtype);
    printf("Base layout: %d\n", sess->base_layout);
    
    // Initialize session
    csinn_session_init(sess);
    
    // No return value for csinn_session_init according to header
    printf("Session initialized\n");
    
    csinn_free_session(sess);
}

int main(int argc, char **argv)
{
    init_testsuite("Testing Conv2D with C920 API\n");
    
    test_check_session_setup();
    test_conv2d_with_proper_api();
    
    // Also run the original problematic test with corrected API
    printf("\n\n=== Running original test with C920 API ===\n");
    
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
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    params->base.api = CSINN_C920;  // Correct API
    
    input->data = conv2d_im2col_fp32_in;
    kernel->data = conv2d_im2col_fp32_ker1;  // Already reordered
    bias->data = conv2d_im2col_fp32_bias;
    
    float output_data[380];
    output->data = output_data;
    
    // Initialize and compute
    int ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("Init returned: %d\n", ret);
    
    if (ret == CSINN_TRUE && params->base.cb && params->base.cb->exec) {
        ret = csinn_conv2d(input, output, kernel, bias, params);
        printf("Compute returned: %d\n", ret);
        
        // Evaluate
        evaluate_error(output->data, conv2d_im2col_fp32_out, 380, CSINN_DTYPE_FLOAT32);
    } else {
        printf("Conv2d not properly initialized, skipping compute\n");
    }
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    
    return done_testing();
}
