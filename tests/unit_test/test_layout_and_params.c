/*
 * Test different layouts and parameter combinations
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>

const char* get_layout_name(int layout) {
    switch(layout) {
        case CSINN_LAYOUT_NCHW: return "NCHW";
        case CSINN_LAYOUT_NHWC: return "NHWC";
        case CSINN_LAYOUT_NC1HWC0: return "NC1HWC0";
        case CSINN_LAYOUT_OIHW: return "OIHW";
        case CSINN_LAYOUT_OHWI: return "OHWI";
        default: return "UNKNOWN";
    }
}

const char* get_api_name(int api) {
    switch(api) {
        case CSINN_REF: return "REF";
        case CSINN_C920: return "C920";
        case CSINN_RVV: return "RVV";
        default: return "UNKNOWN";
    }
}

static void test_different_configurations()
{
    printf("=== Testing different layout and API combinations ===\n\n");
    
    // Test data dimensions
    int in_c = 3, in_h = 4, in_w = 5;
    int out_c = 19, k_h = 3, k_w = 3;
    int out_h = in_h; // with padding=1, stride=1
    int out_w = in_w;
    
    // Input layouts to test
    int input_layouts[] = {CSINN_LAYOUT_NCHW, CSINN_LAYOUT_NHWC, CSINN_LAYOUT_NC1HWC0};
    int kernel_layouts[] = {CSINN_LAYOUT_OIHW, CSINN_LAYOUT_OHWI};
    int apis[] = {CSINN_C920, CSINN_RVV, CSINN_REF};
    
    for (int api_idx = 0; api_idx < 3; api_idx++) {
        for (int in_layout_idx = 0; in_layout_idx < 3; in_layout_idx++) {
            for (int ker_layout_idx = 0; ker_layout_idx < 2; ker_layout_idx++) {
                int api = apis[api_idx];
                int in_layout = input_layouts[in_layout_idx];
                int ker_layout = kernel_layouts[ker_layout_idx];
                
                printf("Testing API=%s, Input Layout=%s, Kernel Layout=%s: ",
                       get_api_name(api), get_layout_name(in_layout), get_layout_name(ker_layout));
                
                struct csinn_tensor *input = csinn_alloc_tensor(NULL);
                input->dim[0] = 1;
                input->dim[1] = in_c;
                input->dim[2] = in_h;
                input->dim[3] = in_w;
                input->dim_count = 4;
                input->dtype = CSINN_DTYPE_FLOAT32;
                input->layout = in_layout;
                
                struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
                kernel->dim[0] = out_c;
                kernel->dim[1] = in_c;
                kernel->dim[2] = k_h;
                kernel->dim[3] = k_w;
                kernel->dim_count = 4;
                kernel->dtype = CSINN_DTYPE_FLOAT32;
                kernel->layout = ker_layout;
                
                struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
                bias->dim[0] = out_c;
                bias->dim_count = 1;
                bias->dtype = CSINN_DTYPE_FLOAT32;
                bias->layout = CSINN_LAYOUT_O;
                
                struct csinn_tensor *output = csinn_alloc_tensor(NULL);
                output->dim[0] = 1;
                output->dim[1] = out_c;
                output->dim[2] = out_h;
                output->dim[3] = out_w;
                output->dim_count = 4;
                output->dtype = CSINN_DTYPE_FLOAT32;
                output->layout = in_layout; // Usually same as input
                
                struct csinn_conv2d_params *params = 
                    csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
                params->base.api = api;
                params->base.layout = in_layout; // Set base layout
                params->stride_height = 1;
                params->stride_width = 1;
                params->pad_left = 1;
                params->pad_right = 1;
                params->pad_top = 1;
                params->pad_down = 1;
                params->group = 1;
                params->dilation_height = 1;
                params->dilation_width = 1;
                
                int ret = csinn_conv2d_init(input, output, kernel, bias, params);
                printf("init=%d, callback=%s\n", ret, 
                       (params->base.cb && params->base.cb->exec) ? "SET" : "NULL");
                
                csinn_free_tensor(input);
                csinn_free_tensor(output);
                csinn_free_tensor(kernel);
                csinn_free_tensor(bias);
                csinn_free_params(params);
            }
        }
    }
}

static void test_direct_rvv_with_debug()
{
    printf("\n=== Testing direct RVV call with debug info ===\n");
    
    // Create simple test case
    int in_size = 1 * 1 * 3 * 3;  // 1x1x3x3
    int out_size = 1 * 1 * 3 * 3; // 1x1x3x3
    int kernel_size = 1 * 1 * 3 * 3; // 1x1x3x3
    
    float input_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel_data[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0}; // Identity-like
    float bias_data[1] = {0.5};
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
    params->dilation_height = 1;
    params->dilation_width = 1;
    
    // Print kernel before reorder
    printf("Kernel before reorder:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", kernel_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Reorder kernel
    shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
    
    printf("\nKernel after reorder:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", kernel_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Clear output
    memset(output_data, 0, sizeof(output_data));
    
    // Direct RVV call
    int ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
    printf("\nDirect RVV returned: %d\n", ret);
    
    printf("Output:\n");
    for (int i = 0; i < 9; i++) {
        printf("%.1f ", output_data[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    
    // Expected output for identity kernel with padding=1, stride=1:
    // Should be centered 3x3 of input + bias
    printf("\nExpected (approximately):\n");
    printf("0.5 0.5 0.5\n");
    printf("0.5 5.5 0.5\n");  // center should be input[4] + bias = 5 + 0.5
    printf("0.5 0.5 0.5\n");
    
    csinn_free_tensor(input);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

static void test_memory_alignment()
{
    printf("\n=== Testing memory alignment requirements ===\n");
    
    // Allocate with different alignments
    float* aligned16 = (float*)shl_mem_alloc(100 * sizeof(float));
    float* aligned32 = (float*)shl_mem_alloc(100 * sizeof(float));
    float* aligned64 = (float*)shl_mem_alloc(100 * sizeof(float));
    
    printf("Default allocation alignment:\n");
    printf("  addr1 %% 16 = %lu\n", (uintptr_t)aligned16 % 16);
    printf("  addr2 %% 32 = %lu\n", (uintptr_t)aligned32 % 32);
    printf("  addr3 %% 64 = %lu\n", (uintptr_t)aligned64 % 64);
    
    shl_mem_free(aligned16);
    shl_mem_free(aligned32);
    shl_mem_free(aligned64);
}

int main(int argc, char **argv)
{
    init_testsuite("Testing different configurations and debugging RVV conv2d.\n");
    
    test_different_configurations();
    test_direct_rvv_with_debug();
    test_memory_alignment();
    
    return done_testing();
}
