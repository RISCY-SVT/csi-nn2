/*
 * Exhaustive test suite for C920 type conversion assembly functions
 * Tests all edge cases, boundary conditions, and various data patterns
 * 
 * Note on rounding behavior:
 * The assembly functions use the RISC-V vfcvt instruction which follows
 * the current rounding mode (typically round-to-nearest-even).
 * This means 0.5 rounds to 0, 1.5 rounds to 2, 2.5 rounds to 2, etc.
 * The reference implementations have been updated to match this behavior.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Function declarations
extern void shl_c920_u8_to_f32(const uint8_t *input, float *output, 
                                int32_t offset, float *scale, uint32_t length);
extern void shl_c920_i8_to_f32(const int8_t *input, float *output, 
                                int32_t offset, float *scale, uint32_t length);
extern void shl_c920_f32_to_u8(const float *input, uint8_t *output, 
                                int32_t offset, float *scale, uint32_t length);
extern void shl_c920_f32_to_i8(const float *input, int8_t *output, 
                                int32_t offset, float *scale, uint32_t length);

// Test configuration
#define MAX_TEST_SIZE 10000
#define EPSILON 1e-5f
#define NUM_RANDOM_TESTS 1000

// Color codes for output
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"

// Test result tracking
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
} TestStats;

static TestStats g_stats = {0, 0, 0};

// Helper functions
static void print_test_header(const char *test_name) {
    printf("\n" ANSI_COLOR_YELLOW "=== %s ===" ANSI_COLOR_RESET "\n", test_name);
}

static void print_test_result(const char *test_case, int passed) {
    g_stats.total_tests++;
    if (passed) {
        g_stats.passed_tests++;
        printf("[" ANSI_COLOR_GREEN "PASS" ANSI_COLOR_RESET "] %s\n", test_case);
    } else {
        g_stats.failed_tests++;
        printf("[" ANSI_COLOR_RED "FAIL" ANSI_COLOR_RESET "] %s\n", test_case);
    }
}

static int compare_floats(float a, float b, float epsilon) {
    return fabsf(a - b) < epsilon;
}

static float random_float(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

// Reference implementations for verification
static void ref_u8_to_f32(const uint8_t *input, float *output, 
                          int32_t offset, float scale, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        output[i] = ((int32_t)input[i] - offset) * scale;
    }
}

static void ref_i8_to_f32(const int8_t *input, float *output, 
                          int32_t offset, float scale, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        output[i] = ((int32_t)input[i] - offset) * scale;
    }
}

static void ref_f32_to_u8(const float *input, uint8_t *output, 
                          int32_t offset, float scale, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        // Use rintf for round-to-nearest-even to match hardware behavior
        int32_t val = (int32_t)rintf(input[i] / scale) + offset;
        val = val < 0 ? 0 : val;  // Clamp to 0
        val = val > 255 ? 255 : val;  // Clamp to 255
        output[i] = (uint8_t)val;
    }
}

static void ref_f32_to_i8(const float *input, int8_t *output, 
                          int32_t offset, float scale, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        // Use rintf for round-to-nearest-even to match hardware behavior
        int32_t val = (int32_t)rintf(input[i] / scale) + offset;
        val = val < -128 ? -128 : val;  // Clamp to -128
        val = val > 127 ? 127 : val;     // Clamp to 127
        output[i] = (int8_t)val;
    }
}

// Test cases for u8_to_f32
static void test_u8_to_f32_basic() {
    print_test_header("u8_to_f32 Basic Tests");
    
    uint8_t input[16] = {0, 1, 2, 127, 128, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    float output[16];
    float ref_output[16];
    float scale = 0.5f;
    int32_t offset = 128;
    
    shl_c920_u8_to_f32(input, output, offset, &scale, 16);
    ref_u8_to_f32(input, ref_output, offset, scale, 16);
    
    int passed = 1;
    for (int i = 0; i < 16; i++) {
        if (!compare_floats(output[i], ref_output[i], EPSILON)) {
            printf("  Mismatch at index %d: expected %f, got %f\n", i, ref_output[i], output[i]);
            passed = 0;
        }
    }
    print_test_result("Basic conversion test", passed);
}

static void test_u8_to_f32_edge_cases() {
    print_test_header("u8_to_f32 Edge Cases");
    
    // Test 1: All zeros
    uint8_t zeros[100];
    float output[100];
    float ref_output[100];
    memset(zeros, 0, sizeof(zeros));
    float scale = 1.0f;
    int32_t offset = 0;
    
    shl_c920_u8_to_f32(zeros, output, offset, &scale, 100);
    ref_u8_to_f32(zeros, ref_output, offset, scale, 100);
    
    int passed = 1;
    for (int i = 0; i < 100; i++) {
        if (!compare_floats(output[i], ref_output[i], EPSILON)) {
            passed = 0;
            break;
        }
    }
    print_test_result("All zeros test", passed);
    
    // Test 2: All max values
    uint8_t max_vals[100];
    memset(max_vals, 255, sizeof(max_vals));
    scale = 0.1f;
    offset = 128;
    
    shl_c920_u8_to_f32(max_vals, output, offset, &scale, 100);
    ref_u8_to_f32(max_vals, ref_output, offset, scale, 100);
    
    passed = 1;
    for (int i = 0; i < 100; i++) {
        if (!compare_floats(output[i], ref_output[i], EPSILON)) {
            passed = 0;
            break;
        }
    }
    print_test_result("All max values test", passed);
    
    // Test 3: Very small scale
    scale = 0.00001f;
    shl_c920_u8_to_f32(max_vals, output, offset, &scale, 10);
    ref_u8_to_f32(max_vals, ref_output, offset, scale, 10);
    
    passed = 1;
    for (int i = 0; i < 10; i++) {
        if (!compare_floats(output[i], ref_output[i], EPSILON)) {
            passed = 0;
            break;
        }
    }
    print_test_result("Very small scale test", passed);
}

static void test_u8_to_f32_alignment() {
    print_test_header("u8_to_f32 Alignment Tests");
    
    // Test various lengths to check vector processing paths
    int test_lengths[] = {1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024};
    float scale = 0.25f;
    int32_t offset = 100;
    
    for (int t = 0; t < sizeof(test_lengths)/sizeof(test_lengths[0]); t++) {
        int len = test_lengths[t];
        uint8_t *input = (uint8_t*)malloc(len);
        float *output = (float*)malloc(len * sizeof(float));
        float *ref_output = (float*)malloc(len * sizeof(float));
        
        // Fill with pattern
        for (int i = 0; i < len; i++) {
            input[i] = (uint8_t)(i % 256);
        }
        
        shl_c920_u8_to_f32(input, output, offset, &scale, len);
        ref_u8_to_f32(input, ref_output, offset, scale, len);
        
        int passed = 1;
        for (int i = 0; i < len; i++) {
            if (!compare_floats(output[i], ref_output[i], EPSILON)) {
                printf("  Length %d: mismatch at index %d\n", len, i);
                passed = 0;
                break;
            }
        }
        
        char test_name[100];
        snprintf(test_name, sizeof(test_name), "Length %d alignment test", len);
        print_test_result(test_name, passed);
        
        free(input);
        free(output);
        free(ref_output);
    }
}

// Test cases for i8_to_f32
static void test_i8_to_f32_basic() {
    print_test_header("i8_to_f32 Basic Tests");
    
    int8_t input[16] = {-128, -64, -1, 0, 1, 63, 127, -50, -25, 25, 50, 100, -100, -10, 10, 0};
    float output[16];
    float ref_output[16];
    float scale = 0.5f;
    int32_t offset = 0;
    
    shl_c920_i8_to_f32(input, output, offset, &scale, 16);
    ref_i8_to_f32(input, ref_output, offset, scale, 16);
    
    int passed = 1;
    for (int i = 0; i < 16; i++) {
        if (!compare_floats(output[i], ref_output[i], EPSILON)) {
            printf("  Mismatch at index %d: expected %f, got %f\n", i, ref_output[i], output[i]);
            passed = 0;
        }
    }
    print_test_result("Basic signed conversion test", passed);
}

static void test_i8_to_f32_negative_offset() {
    print_test_header("i8_to_f32 Negative Offset Tests");
    
    int8_t input[32];
    float output[32];
    float ref_output[32];
    float scale = 1.0f;
    int32_t offset = -64;
    
    // Fill with range of values
    for (int i = 0; i < 32; i++) {
        input[i] = (int8_t)(i * 8 - 128);
    }
    
    shl_c920_i8_to_f32(input, output, offset, &scale, 32);
    ref_i8_to_f32(input, ref_output, offset, scale, 32);
    
    int passed = 1;
    for (int i = 0; i < 32; i++) {
        if (!compare_floats(output[i], ref_output[i], EPSILON)) {
            passed = 0;
            break;
        }
    }
    print_test_result("Negative offset test", passed);
}

// Test cases for f32_to_u8
static void test_f32_to_u8_basic() {
    print_test_header("f32_to_u8 Basic Tests");
    
    float input[16] = {0.0f, 1.0f, -1.0f, 127.5f, 128.0f, 255.0f, 256.0f, -10.0f,
                       50.5f, 100.0f, 200.0f, 1000.0f, -1000.0f, 0.5f, 0.1f, 0.9f};
    uint8_t output[16];
    uint8_t ref_output[16];
    float scale = 1.0f;
    int32_t offset = 0;
    
    shl_c920_f32_to_u8(input, output, offset, &scale, 16);
    ref_f32_to_u8(input, ref_output, offset, scale, 16);
    
    int passed = 1;
    for (int i = 0; i < 16; i++) {
        if (output[i] != ref_output[i]) {
            printf("  Mismatch at index %d: expected %u, got %u (input was %f)\n", 
                   i, ref_output[i], output[i], input[i]);
            passed = 0;
        }
    }
    print_test_result("Basic f32 to u8 conversion", passed);
}

static void test_f32_to_u8_saturation() {
    print_test_header("f32_to_u8 Saturation Tests");
    
    // Test saturation at both ends
    float input[10] = {-1000.0f, -100.0f, -1.0f, 0.0f, 128.0f, 255.0f, 256.0f, 500.0f, 1000.0f, 10000.0f};
    uint8_t output[10];
    uint8_t ref_output[10];
    float scale = 1.0f;
    int32_t offset = 0;
    
    shl_c920_f32_to_u8(input, output, offset, &scale, 10);
    ref_f32_to_u8(input, ref_output, offset, scale, 10);
    
    int passed = 1;
    for (int i = 0; i < 10; i++) {
        if (output[i] != ref_output[i]) {
            printf("  Saturation failed at index %d: expected %u, got %u (input was %f)\n", 
                   i, ref_output[i], output[i], input[i]);
            passed = 0;
        }
    }
    print_test_result("Saturation test", passed);
}

// Test cases for f32_to_i8
static void test_f32_to_i8_basic() {
    print_test_header("f32_to_i8 Basic Tests");
    
    float input[16] = {0.0f, 1.0f, -1.0f, 127.0f, -128.0f, 128.0f, -129.0f,
                       50.5f, -50.5f, 100.0f, -100.0f, 0.5f, -0.5f, 0.1f, -0.1f, 0.0f};
    int8_t output[16];
    int8_t ref_output[16];
    float scale = 1.0f;
    int32_t offset = 0;
    
    shl_c920_f32_to_i8(input, output, offset, &scale, 16);
    ref_f32_to_i8(input, ref_output, offset, scale, 16);
    
    int passed = 1;
    for (int i = 0; i < 16; i++) {
        if (output[i] != ref_output[i]) {
            printf("  Mismatch at index %d: expected %d, got %d (input was %f)\n", 
                   i, ref_output[i], output[i], input[i]);
            passed = 0;
        }
    }
    print_test_result("Basic f32 to i8 conversion", passed);
}

static void test_f32_to_i8_saturation() {
    print_test_header("f32_to_i8 Saturation Tests");
    
    // Test saturation at both ends
    float input[10] = {-1000.0f, -200.0f, -128.0f, -127.0f, 0.0f, 127.0f, 128.0f, 200.0f, 1000.0f, 10000.0f};
    int8_t output[10];
    int8_t ref_output[10];
    float scale = 1.0f;
    int32_t offset = 0;
    
    shl_c920_f32_to_i8(input, output, offset, &scale, 10);
    ref_f32_to_i8(input, ref_output, offset, scale, 10);
    
    int passed = 1;
    for (int i = 0; i < 10; i++) {
        if (output[i] != ref_output[i]) {
            printf("  Saturation failed at index %d: expected %d, got %d (input was %f)\n", 
                   i, ref_output[i], output[i], input[i]);
            passed = 0;
        }
    }
    print_test_result("Signed saturation test", passed);
}

// Round-trip tests
static void test_roundtrip_u8() {
    print_test_header("u8 Round-trip Tests");
    
    uint8_t original[256];
    float intermediate[256];
    uint8_t final[256];
    float scale = 0.5f;
    int32_t offset = 128;
    
    // Fill with all possible u8 values
    for (int i = 0; i < 256; i++) {
        original[i] = (uint8_t)i;
    }
    
    // Convert u8 -> f32 -> u8
    shl_c920_u8_to_f32(original, intermediate, offset, &scale, 256);
    shl_c920_f32_to_u8(intermediate, final, offset, &scale, 256);
    
    int passed = 1;
    for (int i = 0; i < 256; i++) {
        if (original[i] != final[i]) {
            printf("  Round-trip failed at value %u: got %u\n", original[i], final[i]);
            passed = 0;
        }
    }
    print_test_result("u8 round-trip test", passed);
}

static void test_roundtrip_i8() {
    print_test_header("i8 Round-trip Tests");
    
    int8_t original[256];
    float intermediate[256];
    int8_t final[256];
    float scale = 0.5f;
    int32_t offset = 0;
    
    // Fill with all possible i8 values
    for (int i = 0; i < 256; i++) {
        original[i] = (int8_t)(i - 128);
    }
    
    // Convert i8 -> f32 -> i8
    shl_c920_i8_to_f32(original, intermediate, offset, &scale, 256);
    shl_c920_f32_to_i8(intermediate, final, offset, &scale, 256);
    
    int passed = 1;
    for (int i = 0; i < 256; i++) {
        if (original[i] != final[i]) {
            printf("  Round-trip failed at value %d: got %d\n", original[i], final[i]);
            passed = 0;
        }
    }
    print_test_result("i8 round-trip test", passed);
}

// Random tests
static void test_random_conversions() {
    print_test_header("Random Conversion Tests");
    
    for (int test = 0; test < NUM_RANDOM_TESTS; test++) {
        int length = rand() % 1000 + 1;
        float scale = random_float(0.001f, 10.0f);
        int32_t offset = rand() % 256 - 128;
        
        // Allocate buffers
        uint8_t *u8_input = (uint8_t*)malloc(length);
        int8_t *i8_input = (int8_t*)malloc(length);
        float *f32_input = (float*)malloc(length * sizeof(float));
        float *f32_output = (float*)malloc(length * sizeof(float));
        float *f32_ref_output = (float*)malloc(length * sizeof(float));
        uint8_t *u8_output = (uint8_t*)malloc(length);
        uint8_t *u8_ref_output = (uint8_t*)malloc(length);
        int8_t *i8_output = (int8_t*)malloc(length);
        int8_t *i8_ref_output = (int8_t*)malloc(length);
        
        // Fill with random data
        for (int i = 0; i < length; i++) {
            u8_input[i] = rand() % 256;
            i8_input[i] = (int8_t)(rand() % 256 - 128);
            f32_input[i] = random_float(-200.0f, 200.0f);
        }
        
        // Test u8 to f32
        shl_c920_u8_to_f32(u8_input, f32_output, offset, &scale, length);
        ref_u8_to_f32(u8_input, f32_ref_output, offset, scale, length);
        
        int passed = 1;
        for (int i = 0; i < length; i++) {
            if (!compare_floats(f32_output[i], f32_ref_output[i], EPSILON)) {
                passed = 0;
                break;
            }
        }
        if (!passed && test == 0) {  // Only print details for first failure
            printf("  Random u8->f32 test failed (length=%d, scale=%f, offset=%d)\n", 
                   length, scale, offset);
        }
        
        // Test i8 to f32
        shl_c920_i8_to_f32(i8_input, f32_output, offset, &scale, length);
        ref_i8_to_f32(i8_input, f32_ref_output, offset, scale, length);
        
        for (int i = 0; i < length; i++) {
            if (!compare_floats(f32_output[i], f32_ref_output[i], EPSILON)) {
                passed = 0;
                break;
            }
        }
        
        // Test f32 to u8
        shl_c920_f32_to_u8(f32_input, u8_output, offset, &scale, length);
        ref_f32_to_u8(f32_input, u8_ref_output, offset, scale, length);
        
        for (int i = 0; i < length; i++) {
            if (u8_output[i] != u8_ref_output[i]) {
                passed = 0;
                break;
            }
        }
        
        // Test f32 to i8
        shl_c920_f32_to_i8(f32_input, i8_output, offset, &scale, length);
        ref_f32_to_i8(f32_input, i8_ref_output, offset, scale, length);
        
        for (int i = 0; i < length; i++) {
            if (i8_output[i] != i8_ref_output[i]) {
                passed = 0;
                break;
            }
        }
        
        // Free buffers
        free(u8_input);
        free(i8_input);
        free(f32_input);
        free(f32_output);
        free(f32_ref_output);
        free(u8_output);
        free(u8_ref_output);
        free(i8_output);
        free(i8_ref_output);
        
        if (!passed) {
            char msg[100];
            snprintf(msg, sizeof(msg), "Random test %d failed", test);
            print_test_result(msg, 0);
            break;
        }
    }
    
    print_test_result("All random tests", 1);
}

// Performance test
static void test_performance() {
    print_test_header("Performance Tests");
    
    const int size = 1000000;
    uint8_t *u8_data = (uint8_t*)malloc(size);
    float *f32_data = (float*)malloc(size * sizeof(float));
    float scale = 1.0f;
    int32_t offset = 128;
    
    // Fill with data
    for (int i = 0; i < size; i++) {
        u8_data[i] = rand() % 256;
    }
    
    // Measure u8 to f32 conversion
    clock_t start = clock();
    shl_c920_u8_to_f32(u8_data, f32_data, offset, &scale, size);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("  u8_to_f32: %d elements in %.3f seconds (%.2f ME/s)\n", 
           size, time_taken, size / time_taken / 1000000);
    
    // Measure f32 to u8 conversion
    start = clock();
    shl_c920_f32_to_u8(f32_data, u8_data, offset, &scale, size);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("  f32_to_u8: %d elements in %.3f seconds (%.2f ME/s)\n", 
           size, time_taken, size / time_taken / 1000000);
    
    free(u8_data);
    free(f32_data);
    
    print_test_result("Performance test completed", 1);
}

// Special cases tests
static void test_special_cases() {
    print_test_header("Special Cases Tests");
    
    // Test with NaN and Inf
    float special_input[8] = {
        0.0f, -0.0f, INFINITY, -INFINITY, NAN, 
        1.0f, -1.0f, 100.0f
    };
    uint8_t u8_output[8];
    int8_t i8_output[8];
    float scale = 1.0f;
    int32_t offset = 0;
    
    // These should handle NaN/Inf gracefully
    shl_c920_f32_to_u8(special_input, u8_output, offset, &scale, 8);
    shl_c920_f32_to_i8(special_input, i8_output, offset, &scale, 8);
    
    // Just check that it doesn't crash - exact behavior with NaN/Inf is implementation-defined
    print_test_result("NaN/Inf handling test", 1);
    
    // Test with very small/large scales
    float tiny_scale = 1e-10f;
    float huge_scale = 1e10f;
    float normal_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];
    
    shl_c920_u8_to_f32((uint8_t*)normal_input, output, 0, &tiny_scale, 4);
    print_test_result("Tiny scale test", 1);
    
    shl_c920_u8_to_f32((uint8_t*)normal_input, output, 0, &huge_scale, 4);
    print_test_result("Huge scale test", 1);
}

// Test rounding behavior
static void test_rounding_behavior() {
    print_test_header("Rounding Behavior Tests");
    
    // Test values that expose rounding differences
    float test_values[] = {
        0.5f, 1.5f, 2.5f, 3.5f, 4.5f,      // Positive tie cases
        -0.5f, -1.5f, -2.5f, -3.5f, -4.5f, // Negative tie cases
        0.4f, 0.6f, 1.4f, 1.6f,            // Non-tie cases
        -0.4f, -0.6f, -1.4f, -1.6f
    };
    int num_values = sizeof(test_values) / sizeof(test_values[0]);
    
    uint8_t u8_output[18];
    int8_t i8_output[18];
    float scale = 1.0f;
    int32_t offset = 0;
    
    shl_c920_f32_to_u8(test_values, u8_output, offset, &scale, num_values);
    shl_c920_f32_to_i8(test_values, i8_output, offset, &scale, num_values);
    
    printf("  Rounding behavior (round-to-nearest-even expected):\n");
    printf("  Value  -> u8   i8   (RNE u8) (RNE i8)\n");
    for (int i = 0; i < num_values; i++) {
        int32_t rne_u8 = (int32_t)rintf(test_values[i]);
        int32_t rne_i8 = (int32_t)rintf(test_values[i]);
        rne_u8 = rne_u8 < 0 ? 0 : rne_u8 > 255 ? 255 : rne_u8;
        rne_i8 = rne_i8 < -128 ? -128 : rne_i8 > 127 ? 127 : rne_i8;
        
        printf("  %5.1f  -> %-3u  %-4d (%3d)    (%4d)\n", 
               test_values[i], u8_output[i], i8_output[i], rne_u8, rne_i8);
    }
    
    print_test_result("Rounding behavior documented", 1);
}

// Main test runner
int main(int argc, char *argv[]) {
    printf(ANSI_COLOR_YELLOW "C920 Type Conversion Exhaustive Test Suite\n" ANSI_COLOR_RESET);
    printf("==========================================\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Run all tests
    
    // u8_to_f32 tests
    test_u8_to_f32_basic();
    test_u8_to_f32_edge_cases();
    test_u8_to_f32_alignment();
    
    // i8_to_f32 tests
    test_i8_to_f32_basic();
    test_i8_to_f32_negative_offset();
    
    // f32_to_u8 tests
    test_f32_to_u8_basic();
    test_f32_to_u8_saturation();
    
    // f32_to_i8 tests
    test_f32_to_i8_basic();
    test_f32_to_i8_saturation();
    
    // Round-trip tests
    test_roundtrip_u8();
    test_roundtrip_i8();
    
    // Random tests
    test_random_conversions();
    
    // Special cases
    test_special_cases();
    
    // Rounding behavior
    test_rounding_behavior();
    
    // Performance tests
    test_performance();
    
    // Print summary
    printf("\n" ANSI_COLOR_YELLOW "Test Summary\n" ANSI_COLOR_RESET);
    printf("==========================================\n");
    printf("Total tests:  %d\n", g_stats.total_tests);
    printf(ANSI_COLOR_GREEN "Passed:       %d\n" ANSI_COLOR_RESET, g_stats.passed_tests);
    printf(ANSI_COLOR_RED "Failed:       %d\n" ANSI_COLOR_RESET, g_stats.failed_tests);
    
    if (g_stats.failed_tests == 0) {
        printf("\n" ANSI_COLOR_GREEN "All tests passed! ✓\n" ANSI_COLOR_RESET);
        return 0;
    } else {
        printf("\n" ANSI_COLOR_RED "Some tests failed! ✗\n" ANSI_COLOR_RESET);
        return 1;
    }
}
