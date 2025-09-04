// Stub functions for int8 operations (not supported on K1X)
#include "c920v2/c920v2.h"

// Stub for missing int8 convolution init
int shl_c920v2_conv2d_init_int8(struct csinn_tensor *input,
                                 struct csinn_tensor *output,
                                 struct csinn_tensor *kernel,
                                 struct csinn_tensor *bias,
                                 struct csinn_conv2d_params *params) {
    // Return error - int8 not supported
    shl_debug_error("INT8 convolution not supported on K1X\n");
    return CSINN_UNSUPPORT_DTYPE;
}

int shl_c920v2_conv1x1s1_gemm_packn_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                         struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                         struct csinn_conv2d_params *params) {
    // Return error - int8 not supported
    shl_debug_error("INT8 GEMM not supported on K1X\n");
    return CSINN_UNSUPPORT_DTYPE;
}

int shl_c920v2_conv1x1s1_gemm_pack1ton_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params) {
    // Return error - int8 not supported
    shl_debug_error("INT8 GEMM not supported on K1X\n");
    return CSINN_UNSUPPORT_DTYPE;
}

int shl_c920v2_conv1x1s1_gemm_packnto1_int8(struct csinn_tensor *input, struct csinn_tensor *output,
                                            struct csinn_tensor *kernel, struct csinn_tensor *bias,
                                            struct csinn_conv2d_params *params) {
    // Return error - int8 not supported
    shl_debug_error("INT8 GEMM not supported on K1X\n");
    return CSINN_UNSUPPORT_DTYPE;
}

void shl_c920v2_ncxhwx_gemm_12xpackn_int8_dot(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                              int32_t *bias, int m, int k, int n, int32_t out_zp,
                                              int32_t *mult, int32_t *shift) {
    // Return error - int8 not supported
    shl_debug_error("INT8 GEMM not supported on K1X\n");
}

void shl_c920v2_ncxhwx_gemm_4xpack2n_int8(int8_t *dst, const int8_t *sa, const int8_t *sb,
                                          int32_t *bias, int m, int k, int n, int32_t out_zp,
                                          int32_t *mult, int32_t *shift) {
    // Return error - int8 not supported
    shl_debug_error("INT8 GEMM not supported on K1X\n");
}

