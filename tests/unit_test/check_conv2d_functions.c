/*
 * Check available conv2d functions and verify API
 */

#include "csi_nn.h"
#include "test_utils.h"
#include <stdio.h>

// External RVV functions
extern int shl_rvv_conv_im2col_gemm_fp32(struct csinn_tensor *input, struct csinn_tensor *output,
                                          struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                          struct csinn_conv2d_params *params);

extern void shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(struct csinn_tensor *kernel,
                                                          struct csinn_conv2d_params *params);

// External reference functions  
extern int shl_ref_conv2d_f32(struct csinn_tensor *input, struct csinn_tensor *output,
                              struct csinn_tensor *kernel, struct csinn_tensor *bias,
                              struct csinn_conv2d_params *params);

int main()
{
    printf("=== Checking Conv2D Function Availability ===\n\n");
    
    // Check enum values
    printf("API enum values:\n");
    printf("  CSINN_REF = %d\n", CSINN_REF);
    printf("  CSINN_C920 = %d\n", CSINN_C920);
    printf("  CSINN_RVV = %d\n", CSINN_RVV);
    
    printf("\nData type enum values:\n");
    printf("  CSINN_DTYPE_FLOAT32 = %d\n", CSINN_DTYPE_FLOAT32);
    printf("  CSINN_DTYPE_FLOAT16 = %d\n", CSINN_DTYPE_FLOAT16);
    
    printf("\nLayout enum values:\n");
    printf("  CSINN_LAYOUT_NCHW = %d\n", CSINN_LAYOUT_NCHW);
    printf("  CSINN_LAYOUT_OIHW = %d\n", CSINN_LAYOUT_OIHW);
    
    printf("\nStatus enum values:\n");
    printf("  CSINN_TRUE = %d\n", CSINN_TRUE);
    printf("  CSINN_FALSE = %d\n", CSINN_FALSE);
    
    // Test basic initialization
    printf("\n=== Testing basic Conv2D initialization ===\n");
    
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = 1;
    input->dim[2] = 3;
    input->dim[3] = 3;
    input->dim_count = 4;
    input->dtype = CSINN_DTYPE_FLOAT32;
    input->layout = CSINN_LAYOUT_NCHW;
    
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = 1;
    kernel->dim[1] = 1;
    kernel->dim[2] = 3;
    kernel->dim[3] = 3;
    kernel->dim_count = 4;
    kernel->dtype = CSINN_DTYPE_FLOAT32;
    kernel->layout = CSINN_LAYOUT_OIHW;
    
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = 1;
    bias->dim_count = 1;
    bias->dtype = CSINN_DTYPE_FLOAT32;
    bias->layout = CSINN_LAYOUT_O;
    
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
    
    // Test with different APIs
    printf("\nTesting with CSINN_C920 API:\n");
    params->base.api = CSINN_C920;
    int ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  csinn_conv2d_init returned: %d\n", ret);
    printf("  Callback assigned: %s\n", params->base.cb ? "YES" : "NO");
    if (params->base.cb) {
        printf("    init: %s\n", params->base.cb->init ? "SET" : "NULL");
        printf("    exec: %s\n", params->base.cb->exec ? "SET" : "NULL");
        printf("    est: %s\n", params->base.cb->est ? "SET" : "NULL");
        printf("    caps: %s\n", params->base.cb->caps ? "SET" : "NULL");
        printf("    perf: %s\n", params->base.cb->perf ? "SET" : "NULL");
    }
    
    printf("\nTesting with CSINN_REF API:\n");
    params->base.api = CSINN_REF;
    params->base.cb = NULL; // Reset callback
    ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  csinn_conv2d_init returned: %d\n", ret);
    printf("  Callback assigned: %s\n", params->base.cb ? "YES" : "NO");
    
    printf("\nTesting with CSINN_RVV API:\n");
    params->base.api = CSINN_RVV;
    params->base.cb = NULL; // Reset callback
    ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  csinn_conv2d_init returned: %d\n", ret);
    printf("  Callback assigned: %s\n", params->base.cb ? "YES" : "NO");
    
    // Check if direct RVV functions are accessible
    printf("\n=== Checking direct function pointers ===\n");
    printf("shl_rvv_conv_im2col_gemm_fp32: %s\n", 
           shl_rvv_conv_im2col_gemm_fp32 ? "AVAILABLE" : "NOT FOUND");
    printf("shl_rvv_conv_im2col_gemm_reorder_kernel_fp32: %s\n",
           shl_rvv_conv_im2col_gemm_reorder_kernel_fp32 ? "AVAILABLE" : "NOT FOUND");
    printf("shl_ref_conv2d_f32: %s\n",
           shl_ref_conv2d_f32 ? "AVAILABLE" : "NOT FOUND");
    
    // Test direct call
    if (shl_rvv_conv_im2col_gemm_fp32) {
        printf("\nTesting direct RVV call (with dummy data):\n");
        float dummy_in[9] = {1,2,3,4,5,6,7,8,9};
        float dummy_ker[9] = {0,0,0,0,1,0,0,0,0};
        float dummy_bias[1] = {0};
        float dummy_out[9] = {0};
        
        input->data = dummy_in;
        kernel->data = dummy_ker;
        bias->data = dummy_bias;
        output->data = dummy_out;
        
        params->base.api = CSINN_C920;
        
        // First reorder kernel
        shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
        
        // Then compute
        ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
        printf("  Direct RVV call returned: %d\n", ret);
        printf("  Output: ");
        for (int i = 0; i < 9; i++) {
            printf("%.1f ", dummy_out[i]);
        }
        printf("\n");
    }
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
    
    return 0;
}
