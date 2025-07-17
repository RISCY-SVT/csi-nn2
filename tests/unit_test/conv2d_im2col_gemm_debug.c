/*
 * Debug version of conv2d_im2col_gemm test
 * 
 * This enhanced test adds:
 * - Input/output data dumps
 * - Memory pattern verification
 * - Step-by-step validation
 */

#include "./valid_data/conv2d.dat"
#include "csi_nn.h"
#include "rvv/rvv.h"
#include "test_utils.h"
#include <string.h>

#define MAX_MISMATCH 5
#define DEBUG_DUMP 1

static void dump_tensor_info(struct csinn_tensor *t, const char *name)
{
    printf("\n=== Tensor: %s ===\n", name);
    printf("Dims: ");
    for (int i = 0; i < t->dim_count; i++) {
        printf("%d ", t->dim[i]);
    }
    printf("\nDtype: %d, Layout: %d\n", t->dtype, t->layout);
    printf("Total size: %d elements\n", csinn_tensor_size(t));
}

static void dump_float_data(const float *data, int size, const char *name, int max_print)
{
    printf("\n--- %s (first %d of %d) ---\n", name, 
           size < max_print ? size : max_print, size);
    for (int i = 0; i < size && i < max_print; i++) {
        printf("[%3d] = %f\n", i, data[i]);
    }
}

static void check_memory_pattern(const void *data, size_t size, const char *name)
{
    const uint8_t *bytes = (const uint8_t *)data;
    int zeros = 0, ff_pattern = 0, repeating = 0;
    
    for (size_t i = 0; i < size; i++) {
        if (bytes[i] == 0) zeros++;
        if (bytes[i] == 0xFF) ff_pattern++;
    }
    
    printf("\nMemory pattern check for %s:\n", name);
    printf("  Size: %zu bytes\n", size);
    printf("  Zeros: %d (%.1f%%)\n", zeros, 100.0 * zeros / size);
    printf("  0xFF: %d (%.1f%%)\n", ff_pattern, 100.0 * ff_pattern / size);
    
    // Check for uninitialized memory patterns
    if (zeros > size * 0.9) {
        printf("  WARNING: Memory appears to be mostly zeros!\n");
    }
    if (ff_pattern > size * 0.9) {
        printf("  WARNING: Memory appears to be mostly 0xFF!\n");
    }
}

static void dump_mismatch(const float *out, const float *ref, int size, float tol)
{
    int printed = 0;
    float max_diff = 0.0f;
    int max_diff_idx = -1;
    
    for (int i = 0; i < size; i++) {
        float diff = fabsf(out[i] - ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if (diff > tol && printed < MAX_MISMATCH) {
            printf("  idx %d : out=%f  ref=%f  |diff|=%f\n", i, out[i], ref[i], diff);
            ++printed;
        }
    }
    
    if (max_diff_idx >= 0) {
        printf("\nMax difference at idx %d: |diff|=%f\n", max_diff_idx, max_diff);
    }
}

static void verify_conv2d_im2col_compute_debug(
    void *input_data, void *kernel_data, void *bias_data,
    void *ref_data, int (*compute)(), int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w, int k_h, int k_w,
    enum csinn_dtype_enum dtype, const char *test_name)
{
    printf("\n\n========== TEST: %s ==========\n", test_name);
    
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    input->dim[0] = 1;
    input->dim[1] = in_c;
    input->dim[2] = in_h;
    input->dim[3] = in_w;
    input->dim_count = 4;
    input->name = "input";
    input->dtype  = dtype;
    input->layout = CSINN_LAYOUT_NCHW;
    
    struct csinn_tensor *kernel = csinn_alloc_tensor(NULL);
    kernel->dim[0] = out_c;
    kernel->dim[1] = in_c;
    kernel->dim[2] = k_h;
    kernel->dim[3] = k_w;
    kernel->dim_count = 4;
    kernel->name = "kernel";
    kernel->dtype  = dtype;
    kernel->layout = CSINN_LAYOUT_OIHW;
    
    struct csinn_tensor *bias = csinn_alloc_tensor(NULL);
    bias->dim[0] = out_c;
    bias->dim_count = 1;
    bias->name = "bias";
    bias->dtype  = dtype;
    bias->layout = CSINN_LAYOUT_O;
    
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);
    output->dim[0] = 1;
    output->dim[1] = out_c;
    output->dim[2] = out_h;
    output->dim[3] = out_w;
    output->dim_count = 4;
    output->name = "output";
    output->dtype  = dtype;
    output->layout = CSINN_LAYOUT_NCHW;
    
    int in_size = csinn_tensor_size(input);
    int kernel_size = csinn_tensor_size(kernel);
    int bias_size = csinn_tensor_size(bias);
    int out_size = csinn_tensor_size(output);
    
    printf("\nTensor sizes: input=%d, kernel=%d, bias=%d, output=%d\n",
           in_size, kernel_size, bias_size, out_size);
    
    if (in_size <= 0 || out_size <= 0) {
        printf("ERROR: Invalid tensor size!\n");
        return;
    }
    
    // Dump tensor info
    if (DEBUG_DUMP) {
        dump_tensor_info(input, "input");
        dump_tensor_info(kernel, "kernel");
        dump_tensor_info(bias, "bias");
        dump_tensor_info(output, "output");
    }
    
    struct csinn_conv2d_params *params =
        csinn_alloc_params(sizeof(struct csinn_conv2d_params), NULL);
    params->base.name = "params";
    params->base.api = CSINN_C920;
    params->stride_height = 1;
    params->stride_width = 1;
    params->pad_left = 1;
    params->pad_right = 1;
    params->pad_top = 1;
    params->pad_down = 1;
    params->group = 1;
    
    printf("\nConv params: stride=%dx%d, pad=[%d,%d,%d,%d], group=%d\n",
           params->stride_height, params->stride_width,
           params->pad_top, params->pad_down, 
           params->pad_left, params->pad_right,
           params->group);
    
    input->data  = input_data;
    kernel->data = kernel_data;
    bias->data   = bias_data;
    
    // Allocate and initialize output with pattern
    size_t out_bytes = out_size * sizeof(float);
    output->data = shl_mem_alloc(out_bytes);
    memset(output->data, 0xAA, out_bytes);  // Fill with pattern to detect unwritten areas
    
    // Check input data patterns
    if (DEBUG_DUMP) {
        check_memory_pattern(input_data, in_size * sizeof(float), "input_data");
        check_memory_pattern(kernel_data, kernel_size * sizeof(float), "kernel_data");
        check_memory_pattern(bias_data, bias_size * sizeof(float), "bias_data");
        
        dump_float_data((float*)input_data, in_size, "Input data", 10);
        dump_float_data((float*)kernel_data, kernel_size, "Kernel data", 10);
        dump_float_data((float*)bias_data, bias_size, "Bias data", out_c);
    }
    
    printf("\nCalling compute function...\n");
    int ret = compute(input, output, kernel, bias, params);
    printf("Compute returned: %d\n", ret);
    
    // Check output pattern
    if (DEBUG_DUMP) {
        check_memory_pattern(output->data, out_bytes, "output_data");
        dump_float_data((float*)output->data, out_size, "Output data", 20);
        dump_float_data((float*)ref_data, out_size, "Reference data", 20);
    }
    
    printf("\nEvaluating error metrics:\n");
    evaluate_error(output->data, ref_data, out_size, dtype);
    dump_mismatch((const float *)output->data, (const float *)ref_data, out_size, 1e-3f);
    
    // Additional validation: check for NaN/Inf
    float *out_f = (float*)output->data;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < out_size; i++) {
        if (isnan(out_f[i])) nan_count++;
        if (isinf(out_f[i])) inf_count++;
    }
    if (nan_count > 0) printf("WARNING: Found %d NaN values in output!\n", nan_count);
    if (inf_count > 0) printf("WARNING: Found %d Inf values in output!\n", inf_count);
    
    csinn_free_tensor(input);
    shl_mem_free(output->data);
    csinn_free_tensor(output);
    csinn_free_tensor(kernel);
    csinn_free_tensor(bias);
}

int main(int argc, char **argv)
{
    init_testsuite("Test function of convolution im2col_gemm for (!RVV) REF (DEBUG).\n");
    
    // Test only FP32 compute first to debug
    verify_conv2d_im2col_compute_debug(
        conv2d_im2col_fp32_in, conv2d_im2col_fp32_ker1,
        conv2d_im2col_fp32_bias, conv2d_im2col_fp32_out,
        shl_rvv_conv_im2col_gemm_fp32, 3, 4, 5, 19, 4, 5, 3, 3,
        CSINN_DTYPE_FLOAT32, "FP32 im2col_gemm compute");
    
    return done_testing();
}
