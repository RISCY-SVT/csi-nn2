/*
 * GEMM Tail-N Test for CSI-NN2 RVV Implementation
 * Tests edge cases where n % 8 != 0 for both FP32 and FP16
 * 
 * Build:
 *   riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv0p7_zfh_xtheadc_xtheadvdot \
 *     -mabi=lp64d -mtune=c920 -I../../include -I../.. \
 *     -DRVV_TEST -Wall -Wextra -Werror \
 *     tests/gemm_tail_n_fp32_fp16.c \
 *     source/thead_rvv/fp32/gemm_fp32.c \
 *     source/thead_rvv/fp16/gemm_fp16.c \
 *     source/thead_rvv/reorder.c \
 *     source/utils/debug.c \
 *     -lm -o gemm_tail_test.elf
 * 
 * Run:
 *   qemu-riscv64 -cpu rv64,x-v=true,vlen=128,x-zvfh=true ./gemm_tail_test.elf
 *   # or on real hardware:
 *   ./gemm_tail_test.elf
 * 
 * NOTE: This test should be included in CI pipeline for all GEMM-related PRs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <time.h>

// CSI-NN2 headers
#include "rvv/rvv.h"
#include "csinn/csinn_data_structure.h"
#include "shl_utils.h"

// ANSI color codes for output
#define COLOR_RED    "\x1b[31m"
#define COLOR_GREEN  "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_RESET  "\x1b[0m"

// Static assertions for platform assumptions
_Static_assert(sizeof(float) == 4, "float must be 32-bit");
_Static_assert(sizeof(__fp16) == 2, "half must be 16-bit");

// Error thresholds
#define FP32_MAX_ERROR    1e-4f
#define FP16_MAX_ERROR    1e-2f
#define MIN_COS_SIMILARITY 0.9999f
#define MAX_KL_DIVERGENCE  1e-4f

// RNG seed for reproducibility
#define RANDOM_SEED 123

// Test configuration
typedef struct {
    int m;
    int k;
    int n;
    int use_bias;
    const char *desc;
} test_config_t;

/** DOC: Simple reference GEMM implementation for FP32
 * C = A * B + bias (if bias != NULL)
 * A: m x k (row-major), B: k x n (row-major), C: m x n (row-major)
 * This is the golden reference - simple triple loop
 */
static void ref_gemm_fp32(float *C, const float *A, const float *B,
                          const float *bias, int m, int k, int n, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            if (bias) {
                sum += bias[i];  // Note: bias is per output row in CSI-NN2
            }
            C[i * ldc + j] = sum;
        }
    }
}

/** DOC: Simple reference GEMM implementation for FP16 */
static void ref_gemm_fp16(__fp16 *C, const __fp16 *A, const __fp16 *B,
                          const __fp16 *bias, int m, int k, int n, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;  // Use FP32 accumulator for better precision
            for (int p = 0; p < k; p++) {
                sum += (float)A[i * k + p] * (float)B[p * n + j];
            }
            if (bias) {
                sum += (float)bias[i];
            }
            C[i * ldc + j] = (__fp16)sum;
        }
    }
}

/** DOC: Initialize random generator with fixed seed */
static void init_random(void) {
    srand(RANDOM_SEED);
}

/** DOC: Generate random float in [-1, 1] */
static float random_float(void) {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

/** DOC: Fill float array with random values */
static void fill_random_fp32(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = random_float() * 0.5f;  // Scale down to avoid overflow
    }
}

/** DOC: Fill fp16 array with random values */
static void fill_random_fp16(__fp16 *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (__fp16)(random_float() * 0.5f);
    }
}

/** DOC: Compute cosine similarity between two vectors */
static float cosine_similarity_fp32(const float *a, const float *b, int n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-9f);
}

static float cosine_similarity_fp16(const __fp16 *a, const __fp16 *b, int n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < n; i++) {
        float fa = (float)a[i];
        float fb = (float)b[i];
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-9f);
}

/** DOC: Compute KL divergence (simplified, assumes positive values) */
static float kl_divergence_fp32(const float *p, const float *q, int n) {
    float kl = 0.0f;
    const float epsilon = 1e-9f;
    for (int i = 0; i < n; i++) {
        float pp = fabsf(p[i]) + epsilon;
        float qq = fabsf(q[i]) + epsilon;
        kl += pp * logf(pp / qq);
    }
    return kl / n;
}

static float kl_divergence_fp16(const __fp16 *p, const __fp16 *q, int n) {
    float kl = 0.0f;
    const float epsilon = 1e-9f;
    for (int i = 0; i < n; i++) {
        float pp = fabsf((float)p[i]) + epsilon;
        float qq = fabsf((float)q[i]) + epsilon;
        kl += pp * logf(pp / qq);
    }
    return kl / n;
}

/** DOC: Print hex dump of a matrix row */
static void print_hex_dump_fp32(const char *label, const float *data, int row, int cols) {
    printf("  %s row %d hex: ", label, row);
    const uint32_t *hex_data = (const uint32_t *)data;
    for (int j = 0; j < cols && j < 8; j++) {
        printf("%08x ", hex_data[row * cols + j]);
    }
    if (cols > 8) printf("...");
    printf("\n");
}

static void print_hex_dump_fp16(const char *label, const __fp16 *data, int row, int cols) {
    printf("  %s row %d hex: ", label, row);
    const uint16_t *hex_data = (const uint16_t *)data;
    for (int j = 0; j < cols && j < 16; j++) {
        printf("%04x ", hex_data[row * cols + j]);
    }
    if (cols > 16) printf("...");
    printf("\n");
}

/** DOC: Print first few elements of matrix for debugging */
static void print_matrix_sample_fp32(const char *name, const float *mat, int rows, int cols, int stride) {
    printf("  %s sample [%dx%d]:\n", name, rows, cols);
    for (int i = 0; i < rows && i < 4; i++) {
        printf("    ");
        for (int j = 0; j < cols && j < 8; j++) {
            printf("%7.4f ", mat[i * stride + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 4) printf("    ...\n");
}

/** DOC: Compare two FP32 matrices and report errors */
static int compare_matrices_fp32(const float *out, const float *ref,
                                int m, int n, int ldc_out, int ldc_ref,
                                float max_error, const char *test_name) {
    float max_diff = 0.0f;
    int error_count = 0;
    int first_error_row = -1;
    
    // Flatten for metrics computation
    float *flat_out = malloc(m * n * sizeof(float));
    float *flat_ref = malloc(m * n * sizeof(float));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float out_val = out[i * ldc_out + j];
            float ref_val = ref[i * ldc_ref + j];
            float diff = fabsf(out_val - ref_val);
            
            flat_out[i * n + j] = out_val;
            flat_ref[i * n + j] = ref_val;
            
            if (diff > max_diff) {
                max_diff = diff;
            }
            
            if (diff > max_error) {
                if (error_count < 10) {
                    if (error_count == 0) {
                        printf(COLOR_RED "ERROR in %s:\n" COLOR_RESET, test_name);
                        first_error_row = i;
                    }
                    printf("  [%d,%d]: out=%.6f ref=%.6f diff=%.6e\n",
                           i, j, out_val, ref_val, diff);
                }
                error_count++;
            }
        }
    }
    
    if (error_count > 0) {
        float cos_sim = cosine_similarity_fp32(flat_out, flat_ref, m * n);
        float kl_div = kl_divergence_fp32(flat_out, flat_ref, m * n);
        
        printf("  Total errors: %d/%d (%.2f%%)\n", error_count, m * n, 
               100.0f * error_count / (m * n));
        printf("  Max diff: %.6e (threshold: %.6e)\n", max_diff, max_error);
        printf("  Cosine similarity: %.6f (threshold: %.4f)\n", cos_sim, MIN_COS_SIMILARITY);
        printf("  KL divergence: %.6e (threshold: %.6e)\n", kl_div, MAX_KL_DIVERGENCE);
        
        if (first_error_row >= 0) {
            print_hex_dump_fp32("Output", out, first_error_row, n);
            print_hex_dump_fp32("Reference", ref, first_error_row, n);
        }
        
        printf("  Compiler flags: -march=rv64gcv0p7_zfh_xtheadc_xtheadvdot\n");
        printf("  Kernel: shl_rvv_gemm_8x8_fp32\n");
    }
    
    free(flat_out);
    free(flat_ref);
    
    return error_count;
}

/** DOC: Compare two FP16 matrices and report errors */
static int compare_matrices_fp16(const __fp16 *out, const __fp16 *ref,
                                int m, int n, int ldc_out, int ldc_ref,
                                float max_error, const char *test_name) {
    float max_diff = 0.0f;
    int error_count = 0;
    int first_error_row = -1;
    
    // Flatten for metrics computation
    __fp16 *flat_out = malloc(m * n * sizeof(__fp16));
    __fp16 *flat_ref = malloc(m * n * sizeof(__fp16));
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            __fp16 out_val = out[i * ldc_out + j];
            __fp16 ref_val = ref[i * ldc_ref + j];
            float diff = fabsf((float)out_val - (float)ref_val);
            
            flat_out[i * n + j] = out_val;
            flat_ref[i * n + j] = ref_val;
            
            if (diff > max_diff) {
                max_diff = diff;
            }
            
            if (diff > max_error) {
                if (error_count < 10) {
                    if (error_count == 0) {
                        printf(COLOR_RED "ERROR in %s:\n" COLOR_RESET, test_name);
                        first_error_row = i;
                    }
                    printf("  [%d,%d]: out=%.6f ref=%.6f diff=%.6e\n",
                           i, j, (float)out_val, (float)ref_val, diff);
                }
                error_count++;
            }
        }
    }
    
    if (error_count > 0) {
        float cos_sim = cosine_similarity_fp16(flat_out, flat_ref, m * n);
        float kl_div = kl_divergence_fp16(flat_out, flat_ref, m * n);
        
        printf("  Total errors: %d/%d (%.2f%%)\n", error_count, m * n,
               100.0f * error_count / (m * n));
        printf("  Max diff: %.6e (threshold: %.6e)\n", max_diff, max_error);
        printf("  Cosine similarity: %.6f (threshold: %.4f)\n", cos_sim, MIN_COS_SIMILARITY);
        printf("  KL divergence: %.6e (threshold: %.6e)\n", kl_div, MAX_KL_DIVERGENCE);
        
        if (first_error_row >= 0) {
            print_hex_dump_fp16("Output", out, first_error_row, n);
            print_hex_dump_fp16("Reference", ref, first_error_row, n);
        }
        
        printf("  Compiler flags: -march=rv64gcv0p7_zfh_xtheadc_xtheadvdot\n");
        printf("  Kernel: shl_rvv_gemm_8x16_fp16\n");
    }
    
    free(flat_out);
    free(flat_ref);
    
    return error_count;
}

/** DOC: Test one configuration for FP32 
 * The CSI-NN2 GEMM expects pre-reordered matrices:
 * - Kernel (A) reordered by shl_rvv_reorder_kernel_n8_fp32
 * - Input (B) reordered by shl_rvv_reorder_input_z8_fp32
 */
static int test_gemm_fp32_config(int m, int n, int k, int use_bias) {
    char test_name[128];
    snprintf(test_name, sizeof(test_name), "FP32 GEMM m=%d n=%d k=%d bias=%s",
             m, n, k, use_bias ? "yes" : "no");
    
    // Allocate matrices
    float *A = shl_mem_alloc(m * k * sizeof(float));      // Original kernel matrix
    float *B = shl_mem_alloc(k * n * sizeof(float));      // Original input matrix
    float *A_reordered = shl_mem_alloc(m * k * sizeof(float));  // Reordered kernel
    float *B_reordered = shl_mem_alloc(k * n * sizeof(float));  // Reordered input
    float *C_out = shl_mem_alloc(m * n * sizeof(float));  // Output from RVV
    float *C_ref = shl_mem_alloc(m * n * sizeof(float));  // Reference output
    float *bias = use_bias ? shl_mem_alloc(m * sizeof(float)) : NULL;
    
    // Initialize with random data
    fill_random_fp32(A, m * k);
    fill_random_fp32(B, k * n);
    if (bias) fill_random_fp32(bias, m);
    
    // Clear output matrices
    memset(C_out, 0, m * n * sizeof(float));
    memset(C_ref, 0, m * n * sizeof(float));
    
    // Run reference implementation (uses original layout)
    ref_gemm_fp32(C_ref, A, B, bias, m, k, n, n);
    
    // Reorder matrices for RVV GEMM
    shl_rvv_reorder_kernel_n8_fp32(A, A_reordered, m, k, k);
    shl_rvv_reorder_input_z8_fp32(B, B_reordered, k, n, n);
    
    // Run RVV implementation 
    // TODO: verify with unit test - check if bias indexing is correct
    shl_rvv_gemm_8x8_fp32(C_out, A_reordered, B_reordered, bias, m, k, n, n);
    
    // Debug output for small matrices
    if (m <= 4 && n <= 8 && k <= 4) {
        printf("\n  Debug info for %s:\n", test_name);
        print_matrix_sample_fp32("A (kernel)", A, m, k, k);
        print_matrix_sample_fp32("B (input)", B, k, n, n);
        if (bias) {
            printf("  Bias: ");
            for (int i = 0; i < m && i < 8; i++) printf("%.4f ", bias[i]);
            printf("\n");
        }
        print_matrix_sample_fp32("C_ref", C_ref, m, n, n);
        print_matrix_sample_fp32("C_out", C_out, m, n, n);
    }
    
    // Compare results
    int errors = compare_matrices_fp32(C_out, C_ref, m, n, n, n, FP32_MAX_ERROR, test_name);
    
    // Cleanup
    shl_mem_free(A);
    shl_mem_free(B);
    shl_mem_free(A_reordered);
    shl_mem_free(B_reordered);
    shl_mem_free(C_out);
    shl_mem_free(C_ref);
    if (bias) shl_mem_free(bias);
    
    return errors;
}

/** DOC: Test one configuration for FP16 
 * Note: FP16 GEMM processes 8x16 tiles instead of 8x8
 */
static int test_gemm_fp16_config(int m, int n, int k, int use_bias) {
    char test_name[128];
    snprintf(test_name, sizeof(test_name), "FP16 GEMM m=%d n=%d k=%d bias=%s",
             m, n, k, use_bias ? "yes" : "no");
    
    // Allocate matrices
    __fp16 *A = shl_mem_alloc(m * k * sizeof(__fp16));
    __fp16 *B = shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *A_reordered = shl_mem_alloc(m * k * sizeof(__fp16));
    __fp16 *B_reordered = shl_mem_alloc(k * n * sizeof(__fp16));
    __fp16 *C_out = shl_mem_alloc(m * n * sizeof(__fp16));
    __fp16 *C_ref = shl_mem_alloc(m * n * sizeof(__fp16));
    __fp16 *bias = use_bias ? shl_mem_alloc(m * sizeof(__fp16)) : NULL;
    
    // Initialize with random data
    fill_random_fp16(A, m * k);
    fill_random_fp16(B, k * n);
    if (bias) fill_random_fp16(bias, m);
    
    // Clear output matrices
    memset(C_out, 0, m * n * sizeof(__fp16));
    memset(C_ref, 0, m * n * sizeof(__fp16));
    
    // Run reference implementation
    ref_gemm_fp16(C_ref, A, B, bias, m, k, n, n);
    
    // Reorder matrices for RVV GEMM
    shl_rvv_reorder_kernel_n8_fp16(A, A_reordered, m, k, k);
    shl_rvv_reorder_input_z16_fp16(B, B_reordered, k, n, n);
    
    // Run RVV implementation
    // TODO: verify with unit test - actual kernel processes 8x16 tiles
    shl_rvv_gemm_8x16_fp16(C_out, A_reordered, B_reordered, bias, m, k, n, n);
    
    // Compare results
    int errors = compare_matrices_fp16(C_out, C_ref, m, n, n, n, FP16_MAX_ERROR, test_name);
    
    // Cleanup
    shl_mem_free(A);
    shl_mem_free(B);
    shl_mem_free(A_reordered);
    shl_mem_free(B_reordered);
    shl_mem_free(C_out);
    shl_mem_free(C_ref);
    if (bias) shl_mem_free(bias);
    
    return errors;
}

/** DOC: Test batch of configurations */
static int test_batch(const test_config_t *configs, int num_configs, 
                     int (*test_fp32)(int, int, int, int),
                     int (*test_fp16)(int, int, int, int)) {
    int total_errors = 0;
    int total_tests = 0;
    
    for (int i = 0; i < num_configs; i++) {
        const test_config_t *cfg = &configs[i];
        
        printf(COLOR_YELLOW "\n--- Testing: %s ---\n" COLOR_RESET, cfg->desc);
        
        // Test FP32
        int fp32_errors = test_fp32(cfg->m, cfg->n, cfg->k, cfg->use_bias);
        total_errors += fp32_errors;
        total_tests++;
        if (fp32_errors == 0) {
            printf(COLOR_GREEN "  FP32: PASSED\n" COLOR_RESET);
        }
        
        // Test FP16
        int fp16_errors = test_fp16(cfg->m, cfg->n, cfg->k, cfg->use_bias);
        total_errors += fp16_errors;
        total_tests++;
        if (fp16_errors == 0) {
            printf(COLOR_GREEN "  FP16: PASSED\n" COLOR_RESET);
        }
    }
    
    return total_errors;
}

/** DOC: Main test runner
 * Tests various tail cases where n % 8 != 0 (and n % 16 != 0 for FP16)
 * These are critical for vector kernels that process 8/16 elements at a time
 */
int main(void) {
    init_random();
    // Print timestamp to differentiate logs
    time_t now = time(NULL);
    printf("CSI-NN2 GEMM Tail Test\n");
    printf("======================\n");
    printf("Test started at: %s", ctime(&now));
    printf("Testing GEMM tail cases (n %% 8 != 0) with seed=%d\n", RANDOM_SEED);
    printf("VLEN=%d bits\n", csrr_vlenb() * 8);
    
    // Critical tail cases
    test_config_t tail_configs[] = {
        // Minimal cases to test basic functionality
        {1, 1, 1, 0, "Minimal 1x1x1 no bias"},
        {1, 1, 1, 1, "Minimal 1x1x1 with bias"},
        
        // Single column cases (n=1)
        {1, 7, 1, 0, "Single output column, m=1"},
        {4, 13, 1, 1, "Single output column, m=4"},
        {8, 27, 1, 0, "Single output column, m=8"},
        
        // Small n cases (n < 8)
        {2, 7, 2, 1, "n=2 (quarter tile)"},
        {4, 13, 3, 0, "n=3"},
        {8, 27, 4, 1, "n=4 (half tile)"},
        {12, 55, 5, 0, "n=5"},
        {1, 13, 6, 1, "n=6"},
        {8, 27, 7, 0, "n=7 (almost full tile)"},
        
        // n = 8 boundary case (should work perfectly)
        {4, 13, 8, 1, "n=8 (exactly one tile)"},
        
        // Tail cases (8 < n < 16)
        {2, 7, 9, 0, "n=9 (one tile + 1)"},
        {4, 13, 10, 1, "n=10"},
        {8, 27, 11, 0, "n=11"},
        {12, 55, 13, 1, "n=13"},
        {1, 13, 15, 0, "n=15 (almost two tiles)"},
        
        // n = 16 boundary (important for FP16)
        {4, 13, 16, 1, "n=16 (exactly one FP16 tile)"},
        
        // Larger tail cases
        {8, 27, 17, 0, "n=17 (two tiles + 1)"},
        {4, 13, 23, 1, "n=23 (two tiles + 7)"},
        {12, 55, 31, 0, "n=31 (three tiles + 7)"},
        
        // Edge cases with different m values
        {1, 13, 7, 1, "m=1, n=7 (single row)"},
        {3, 27, 5, 0, "m=3, n=5 (partial m tile)"},
        {7, 55, 3, 1, "m=7, n=3 (partial m tile)"},
        {9, 13, 11, 0, "m=9, n=11 (m and n both have tails)"},
        {15, 27, 13, 1, "m=15, n=13 (larger tails)"},
        
        // Stress test with various k values
        {8, 1, 7, 0, "k=1 (minimal reduction)"},
        {8, 8, 7, 1, "k=8 (one k-block)"},
        {8, 16, 7, 0, "k=16 (two k-blocks)"},
        {8, 31, 7, 1, "k=31 (k with tail)"},
        {8, 64, 7, 0, "k=64 (large k)"},
        {8, 127, 7, 1, "k=127 (large k with tail)"},
    };
    
    int num_configs = sizeof(tail_configs) / sizeof(tail_configs[0]);
    int total_errors = test_batch(tail_configs, num_configs, 
                                 test_gemm_fp32_config, test_gemm_fp16_config);
    int total_tests = num_configs * 2; // FP32 + FP16
    
    // Additional random tests
    printf(COLOR_YELLOW "\n--- Random tail tests ---\n" COLOR_RESET);
    test_config_t random_configs[10];
    for (int i = 0; i < 10; i++) {
        // Generate random dimensions with n having tail
        int n_base = (rand() % 4) * 8;  // 0, 8, 16, 24
        int n_tail = 1 + (rand() % 7);  // 1-7
        
        random_configs[i].m = 1 + (rand() % 32);
        random_configs[i].k = 1 + (rand() % 128);
        random_configs[i].n = n_base + n_tail;
        random_configs[i].use_bias = rand() % 2;
        
        char desc[64];
        snprintf(desc, sizeof(desc), "Random m=%d k=%d n=%d", 
                random_configs[i].m, random_configs[i].k, random_configs[i].n);
        random_configs[i].desc = strdup(desc);
    }
    
    int random_errors = test_batch(random_configs, 10,
                                  test_gemm_fp32_config, test_gemm_fp16_config);
    total_errors += random_errors;
    total_tests += 20; // 10 configs * 2 types
    
    // Free allocated descriptions
    for (int i = 0; i < 10; i++) {
        free((void*)random_configs[i].desc);
    }
    
    // Final summary
    printf("\n================================================\n");
    if (total_errors == 0) {
        printf(COLOR_GREEN "ALL tail-cases PASSED (m<=32, n∉8Z)" COLOR_RESET "\n");
        printf("Total tests: %d\n", total_tests);
        printf("All GEMM tail handling is correct!\n");
    } else {
        printf(COLOR_RED "FAILED: %d errors found" COLOR_RESET "\n", total_errors);
        printf("Total tests: %d\n", total_tests);
        printf("GEMM tail handling needs fixes.\n");
        return 1;
    }
    
    // Additional info about what was tested
    printf("\nTest coverage:\n");
    printf("- Single column outputs (n=1)\n");
    printf("- Small n values (n=2..7)\n");
    printf("- n=8 boundary (exactly one FP32 tile)\n");
    printf("- Medium n values (n=9..15)\n");
    printf("- n=16 boundary (exactly one FP16 tile)\n");
    printf("- Large n values with tails\n");
    printf("- Various m and k dimensions\n");
    printf("- With and without bias\n");
    
    return 0;
}

/* NOTE: This test focuses on edge cases where n is not divisible by 8.
 * Such cases are prone to bugs in vectorized implementations because:
 * 1. The FP32 kernel processes 8 columns at a time (8x8 tiles)
 * 2. The FP16 kernel processes 16 columns at a time (8x16 tiles) 
 * 3. Tail handling code paths are less frequently tested
 * 4. Buffer overruns can occur if bounds checking is incorrect
 * 5. The reorder functions have special handling for tail columns
 * 
 * Known limitations:
 * - This test assumes bias is per output row (not per column)
 * - The test uses row-major layout for reference, while CSI-NN2 uses custom layouts
 * - Error thresholds may need adjustment based on hardware capabilities
 * 
 * The test should be run as part of CI to catch regressions early.
 */
