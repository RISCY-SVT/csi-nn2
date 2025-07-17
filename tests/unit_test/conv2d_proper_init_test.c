/*
 * Test conv2d with proper initialization sequence
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"

static void test_conv2d_with_proper_init(void *input_data, void *kernel_data, void *bias_data,
                                         void *ref_data, int in_c, int in_h, int in_w,
                                         int out_c, int out_h, int out_w, int k_h, int k_w,
                                         enum csinn_dtype_enum dtype, const char *test_name)
{
    printf("\n=== %s ===\n", test_name);
    
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = in_h;
    input->dim[3] = in_w;
    input->dim_count = 4;
    input->dtype = dtype;
    input->layout = CSINN_LAYOUT_NCHW;
    
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = out_c;
    kernel->dim[1] = in_c;
    kernel->dim[2] = k_h;
    kernel->dim[3] = k_w;
    kernel->dim_count = 4;
    kernel->dtype = dtype;
    kernel->layout = CSINN_LAYOUT_OIHW;
    
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = out_c;
    bias->dim_count = 1;
    bias->dtype = dtype;
    bias->layout = CSINN_LAYOUT_O;
    
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_c;
    output->dim[2] = out_h;
    output->dim[3] = out_w;
    output->dim_count = 4;
    output->dtype = dtype;
    output->layout = CSINN_LAYOUT_NCHW;
    
    int out_size = csinn_tensor_size(output);
    
    struct csinn_conv2d_params *params = 
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->base.api = CSINN_C920;  // Set API
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    params->dilation_height = 1;
    params->dilation_width = 1;
    
    // First, let's try initialization with different APIs
    printf("Testing initialization:\n");
    
    // Try C920
    params->base.api = CSINN_C920;
    int ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  CSINN_C920 init: %d, callback: %s\n", ret, 
           (params->base.cb && params->base.cb->exec) ? "SET" : "NOT SET");
    
    // Try REF
    params->base.api = CSINN_REF;
    params->base.cb = NULL;
    ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  CSINN_REF init: %d, callback: %s\n", ret,
           (params->base.cb && params->base.cb->exec) ? "SET" : "NOT SET");
    
    // Try RVV
    params->base.api = CSINN_RVV;
    params->base.cb = NULL;
    ret = csinn_conv2d_init(input, output, kernel, bias, params);
    printf("  CSINN_RVV init: %d, callback: %s\n", ret,
           (params->base.cb && params->base.cb->exec) ? "SET" : "NOT SET");
    
    // Now run the test with the best available option
    printf("\nRunning computation:\n");
    
    // Reset to C920 for actual test
    params->base.api = CSINN_C920;
    params->base.cb = NULL;
    
    input->data = input_data;
    kernel->data = kernel_data;
    bias->data = bias_data;
    output->data = shl_mem_alloc(out_size * sizeof(float));
    
    ret = csinn_conv2d_init(input, output, kernel, bias, params);
    if (ret == CSINN_TRUE && params->base.cb && params->base.cb->exec) {
        printf("Using initialized callback\n");
        ret = csinn_conv2d(input, output, kernel, bias, params);
        printf("Conv2d returned: %d\n", ret);
    } else {
        printf("Init failed or no callback, trying direct RVV call\n");
        if (dtype == CSINN_DTYPE_FLOAT32) {
            // First reorder kernel
            shl_rvv_conv_im2col_gemm_reorder_kernel_fp32(kernel, params);
            // Then compute
            ret = shl_rvv_conv_im2col_gemm_fp32(input, output, kernel, bias, params);
            printf("Direct RVV call returned: %d\n", ret);
        } else if (dtype == CSINN_DTYPE_FLOAT16) {
            shl_rvv_conv_im2col_gemm_reorder_kernel_fp16(kernel, params);
            ret = shl_rvv_conv_im2col_gemm_fp16(input, output, kernel, bias, params);
            printf("Direct RVV call returned: %d\n", ret);
        }
    }
    
    // Evaluate results
    evaluate_error(output->data, ref_data, out_size, dtype);
    
    csinn_free_tensor(input);
    shl_mem_free(output->data);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
    csinn_free_params(params);
}

int main(int argc, char **argv)
{
    init_testsuite("Test conv2d with proper initialization.\n");
    
    // Test FP32
    test_conv2d_with_proper_init(
        conv2d_im2col_fp32_in, conv2d_im2col_fp32_ker1,
        conv2d_im2col_fp32_bias, conv2d_im2col_fp32_out,
        3, 4, 5, 19, 4, 5, 3, 3,
        CSINN_DTYPE_FLOAT32, "FP32 Conv2d Test");
    
    // Test FP16
    test_conv2d_with_proper_init(
        conv2d_im2col_fp16_in, conv2d_im2col_fp16_ker1,
        conv2d_im2col_fp16_bias, conv2d_im2col_fp16_out,
        3, 4, 5, 19, 4, 5, 3, 3,
        CSINN_DTYPE_FLOAT16, "FP16 Conv2d Test");
    
    return done_testing();
}
