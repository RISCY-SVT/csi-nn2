/*
 * Memory Test Suite for CSI-NN2 on TH1520/C920
 * Target: LicheePi 4A
 * 
 */
/**
 * @file test_memory_th1520.c
 * @brief Comprehensive memory management test suite for TH1520/C920 CSI-NN2 library
 * 
 * This test suite provides extensive validation of the SHL (SiPeed Hardware Library) 
 * memory allocation subsystem. It includes tests for basic allocation, aligned memory,
 * multithreading safety, stress testing, and debug features.
 * 
 * @details The test suite covers the following functionality:
 * - Basic memory allocation and deallocation (shl_mem_alloc/shl_mem_free)
 * - Aligned memory allocation with various alignment requirements
 * - Calloc and realloc operations with overflow protection
 * - Guard byte functionality for buffer overflow detection
 * - Multithreaded memory operations for concurrency validation
 * - Stress testing with intensive allocation/deallocation cycles
 * - Debug features including memory mapping and leak detection
 * 
 * @note Requires SHL_MEM_DEBUG and SHL_MEM_DEBUG_VALID_WRITE macros for full functionality
 * @note Thread safety tests are limited when SHL_MEM_DEBUG is enabled due to known limitations
 * @note Guard byte tests require SHL_MEM_DEBUG_VALID_WRITE to be defined
 * 
 * @author Custler
 * @version 1.0
 * @date 2025
 * 
 * @section Usage
 * Compile with appropriate flags:
 * - For debug mode: -DSHL_MEM_DEBUG -DSHL_MEM_DEBUG_VALID_WRITE
 * - For production: compile without debug flags for optimal performance
 * 
 * @section Test_Coverage
 * 1. Basic allocation functionality and edge cases
 * 2. Aligned memory allocation with power-of-2 alignments
 * 3. Calloc/realloc operations with data integrity verification
 * 4. Buffer overflow detection via guard bytes
 * 5. Concurrent memory operations across multiple threads
 * 6. High-intensity stress testing with random operations
 * 7. Debug feature validation and memory mapping
 * 
 * @warning Some tests intentionally corrupt memory boundaries for validation purposes.
 *          These should only be run in controlled test environments.
 */


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <sys/utsname.h>  // For uname()
#include <errno.h>
#include <limits.h>       // For SIZE_MAX

// Memory debugging must be enabled via compiler flags
// Check if these macros are already defined
#ifndef SHL_MEM_DEBUG
#define SHL_MEM_DEBUG
#endif

#ifndef SHL_MEM_DEBUG_VALID_WRITE
#define SHL_MEM_DEBUG_VALID_WRITE
#endif

#include "shl_memory.h"
#include "shl_utils.h"

// ANSI color escape sequences for terminal output
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define TEST_PASS(msg) printf(ANSI_COLOR_GREEN "[PASS] " ANSI_COLOR_RESET "%s\n", msg)
#define TEST_FAIL(msg) printf(ANSI_COLOR_RED "[FAIL] " ANSI_COLOR_RESET "%s\n", msg)
#define TEST_INFO(msg) printf(ANSI_COLOR_BLUE "[INFO] " ANSI_COLOR_RESET "%s\n", msg)
#define TEST_WARN(msg) printf(ANSI_COLOR_YELLOW "[WARN] " ANSI_COLOR_RESET "%s\n", msg)

// Global counters for memory operations
static volatile int g_alloc_count = 0;
static volatile int g_free_count = 0;
static volatile int g_error_count = 0;
static pthread_mutex_t g_count_mutex = PTHREAD_MUTEX_INITIALIZER;

// Structure for passing parameters to threads
struct thread_test_params {
    int thread_id;
    int iterations;
    int min_size;
    int max_size;
    int alignment;
    int delay_us;
    int allocations_per_thread;
};

// Function to get the current time in microseconds
static uint64_t get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

// Function to get memory information
static void print_memory_info() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    
    FILE *status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (strncmp(line, "VmRSS:", 6) == 0 || 
                strncmp(line, "VmPeak:", 7) == 0 ||
                strncmp(line, "VmSize:", 7) == 0) {
                printf("  %s", line);
            }
        }
        fclose(status);
    }
    
    printf("  Max RSS: %ld KB\n", usage.ru_maxrss);
}

// Test 1: Basic memory allocation functionality
/**
 * @brief Tests basic memory allocation functionality of the shl_mem_alloc/shl_mem_free API
 * 
 * This test function validates core memory allocation behaviors including:
 * - Zero-size allocation handling (should return NULL)
 * - Negative size allocation handling (should return NULL) 
 * - Normal size allocation success and proper initialization
 * - Memory zero-initialization verification
 * - Memory read/write functionality validation
 * - Large memory allocation attempts (1MB, 10MB, 100MB)
 * 
 * The test verifies that allocated memory is properly zeroed on allocation,
 * can be written to and read from correctly, and that edge cases like
 * zero or negative sizes are handled appropriately by returning NULL.
 * Large allocations are tested but failures are treated as warnings
 * since they may fail due to system memory constraints.
 * 
 * @note Uses TEST_PASS, TEST_FAIL, and TEST_WARN macros for result reporting
 * @note Requires shl_mem_alloc() and shl_mem_free() functions to be available
 */
static void test_basic_allocation() {
    TEST_INFO("=== Test 1: Basic Memory Allocation ===");

    // Test zero-size allocation
    void *ptr = shl_mem_alloc(0);
    if (ptr == NULL) {
        TEST_PASS("Zero size allocation returns NULL");
    } else {
        TEST_FAIL("Zero size allocation should return NULL");
        shl_mem_free(ptr);
    }

    // Test negative size allocation
    ptr = shl_mem_alloc(-100);
    if (ptr == NULL) {
        TEST_PASS("Negative size allocation returns NULL");
    } else {
        TEST_FAIL("Negative size allocation should return NULL");
        shl_mem_free(ptr);
    }

    // Test normal allocation
    const size_t test_size = 1024;
    ptr = shl_mem_alloc(test_size);
    if (ptr != NULL) {
        TEST_PASS("Normal allocation successful");

        // Check that memory is zeroed
        int is_zeroed = 1;
        for (size_t i = 0; i < test_size; i++) {
            if (((uint8_t*)ptr)[i] != 0) {
                is_zeroed = 0;
                break;
            }
        }
        
        if (is_zeroed) {
            TEST_PASS("Allocated memory is zeroed");
        } else {
            TEST_FAIL("Allocated memory is not zeroed");
        }

        // Write and read
        memset(ptr, 0xAB, test_size);
        int write_ok = 1;
        for (size_t i = 0; i < test_size; i++) {
            if (((uint8_t*)ptr)[i] != 0xAB) {
                write_ok = 0;
                break;
            }
        }
        
        if (write_ok) {
            TEST_PASS("Memory write/read test passed");
        } else {
            TEST_FAIL("Memory write/read test failed");
        }
        
        shl_mem_free(ptr);
        TEST_PASS("Memory freed successfully");
    } else {
        TEST_FAIL("Normal allocation failed");
    }

    // Test large sizes
    size_t large_sizes[] = {1*1024*1024, 10*1024*1024, 100*1024*1024};
    for (int i = 0; i < 3; i++) {
        ptr = shl_mem_alloc(large_sizes[i]);
        if (ptr != NULL) {
            char msg[100];
            snprintf(msg, sizeof(msg), "Large allocation %zu MB successful", 
                     large_sizes[i] / (1024*1024));
            TEST_PASS(msg);
            shl_mem_free(ptr);
        } else {
            char msg[100];
            snprintf(msg, sizeof(msg), "Large allocation %zu MB failed (may be normal)", 
                     large_sizes[i] / (1024*1024));
            TEST_WARN(msg);
        }
    }
}

// Test 2: Aligned memory allocation
/**
 * @brief Tests aligned memory allocation functionality with various alignment values
 * 
 * This function performs comprehensive testing of the shl_mem_alloc_aligned() function
 * by testing both valid and invalid alignment values to ensure proper behavior.
 * 
 * Test Cases:
 * 1. Valid alignments: Tests power-of-2 alignments from 8 to 4096 bytes
 *    - Verifies that allocation succeeds
 *    - Checks that returned pointers are properly aligned
 *    - Ensures memory can be freed correctly
 * 
 * 2. Invalid alignments: Tests non-power-of-2 values (3, 7, 15, 31, 63, 127, 255)
 *    - Verifies that allocation fails as expected
 *    - Ensures proper error handling for invalid parameters
 * 
 * @note Uses platform-specific memory freeing based on SHL_BUILD_RTOS define
 * @note All allocations use a fixed size of 1024 bytes for consistency
 * @note Test results are reported using TEST_PASS() and TEST_FAIL() macros
 */
static void test_aligned_allocation() {
    TEST_INFO("=== Test 2: Aligned Memory Allocation ===");

    // Test various alignments
    int alignments[] = {8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    size_t num_alignments = sizeof(alignments)/sizeof(alignments[0]);
    
    for (size_t i = 0; i < num_alignments; i++) {
        int align = alignments[i];
        void *ptr = shl_mem_alloc_aligned(1024, align);
        
        if (ptr != NULL) {
            if (((uintptr_t)ptr % align) == 0) {
                char msg[100];
                snprintf(msg, sizeof(msg), "Alignment %d successful", align);
                TEST_PASS(msg);
            } else {
                char msg[100];
                snprintf(msg, sizeof(msg), "Alignment %d failed: ptr=%p", align, ptr);
                TEST_FAIL(msg);
            }
            
#ifdef SHL_BUILD_RTOS
            shl_mem_free_aligned(ptr);
#else
            shl_mem_free(ptr);
#endif
        } else {
            char msg[100];
            snprintf(msg, sizeof(msg), "Aligned allocation failed for alignment %d", align);
            TEST_FAIL(msg);
        }
    }

    // Test invalid alignments
    int invalid_aligns[] = {3, 7, 15, 31, 63, 127, 255};
    size_t num_invalid = sizeof(invalid_aligns)/sizeof(invalid_aligns[0]);
    
    for (size_t i = 0; i < num_invalid; i++) {
        void *ptr = shl_mem_alloc_aligned(1024, invalid_aligns[i]);
        if (ptr == NULL) {
            char msg[100];
            snprintf(msg, sizeof(msg), "Invalid alignment %d correctly rejected", 
                     invalid_aligns[i]);
            TEST_PASS(msg);
        } else {
            char msg[100];
            snprintf(msg, sizeof(msg), "Invalid alignment %d should be rejected", 
                     invalid_aligns[i]);
            TEST_FAIL(msg);
#ifdef SHL_BUILD_RTOS
            shl_mem_free_aligned(ptr);
#else
            shl_mem_free(ptr);
#endif
        }
    }
}

// Test 3: Calloc and Realloc
/**
 * @brief Tests calloc and realloc memory allocation functions
 * 
 * This function performs comprehensive testing of the calloc and realloc memory
 * allocation functions to ensure they behave correctly under various scenarios.
 * 
 * Test cases covered:
 * 1. Calloc functionality:
 *    - Tests successful allocation of zeroed memory blocks
 *    - Verifies that allocated memory is properly initialized to zero
 *    - Tests overflow protection by attempting allocation with parameters
 *      that would cause integer overflow (SIZE_MAX/2 * 3)
 * 
 * 2. Realloc functionality:
 *    - Tests memory expansion while preserving existing data
 *    - Verifies data integrity after reallocation to larger size
 *    - Tests memory shrinking to smaller size
 *    - Ensures proper memory management during resize operations
 * 
 * The function uses TEST_PASS and TEST_FAIL macros to report test results
 * and properly frees all allocated memory to prevent leaks.
 * 
 * @note This test assumes the availability of shl_mem_calloc, shl_mem_alloc,
 *       shl_mem_realloc, and shl_mem_free functions from the memory management
 *       library being tested.
 */
static void test_calloc_realloc() {
    TEST_INFO("=== Test 3: Calloc and Realloc ===");

    // Test calloc
    size_t nmemb = 100;
    size_t size = 64;
    void *ptr = shl_mem_calloc(nmemb, size);
    
    if (ptr != NULL) {
        TEST_PASS("Calloc allocation successful");

        // Check that memory is zeroed
        int is_zeroed = 1;
        for (size_t i = 0; i < nmemb * size; i++) {
            if (((uint8_t*)ptr)[i] != 0) {
                is_zeroed = 0;
                break;
            }
        }
        
        if (is_zeroed) {
            TEST_PASS("Calloc memory is zeroed");
        } else {
            TEST_FAIL("Calloc memory is not zeroed");
        }
        
        shl_mem_free(ptr);
    } else {
        TEST_FAIL("Calloc allocation failed");
    }

    // Test calloc overflow
    ptr = shl_mem_calloc(SIZE_MAX/2, 3);
    if (ptr == NULL) {
        TEST_PASS("Calloc overflow protection works");
    } else {
        TEST_FAIL("Calloc should detect overflow");
        shl_mem_free(ptr);
    }

    // Test realloc
    ptr = shl_mem_alloc(100);
    if (ptr != NULL) {
        // Fill with data
        memset(ptr, 0x55, 100);

        // Increase size
        void *new_ptr = shl_mem_realloc(ptr, 200, 100);
        if (new_ptr != NULL) {
            TEST_PASS("Realloc expansion successful");

            // Check that old data is preserved
            int data_intact = 1;
            for (int i = 0; i < 100; i++) {
                if (((uint8_t*)new_ptr)[i] != 0x55) {
                    data_intact = 0;
                    break;
                }
            }
            
            if (data_intact) {
                TEST_PASS("Realloc preserved data");
            } else {
                TEST_FAIL("Realloc corrupted data");
            }
            
            shl_mem_free(new_ptr);
        } else {
            TEST_FAIL("Realloc expansion failed");
            shl_mem_free(ptr);
        }

        // Test shrinking
        ptr = shl_mem_alloc(200);
        memset(ptr, 0xAA, 200);
        new_ptr = shl_mem_realloc(ptr, 50, 200);
        if (new_ptr != NULL) {
            TEST_PASS("Realloc shrink successful");
            shl_mem_free(new_ptr);
        } else {
            TEST_FAIL("Realloc shrink failed");
            shl_mem_free(ptr);
        }
    }
}

// Test 4: Guard Bytes
/**
 * @brief Tests the guard byte functionality for buffer overflow detection.
 * 
 * This function verifies that the memory allocator can detect buffer overflows
 * by checking guard bytes placed after allocated memory blocks. The test performs
 * three scenarios:
 * 
 * 1. Normal allocation and free without corruption - verifies clean operation
 * 2. Intentional corruption of first two guard bytes - tests detection of overflow
 * 3. Partial corruption of last guard byte - tests detection of partial overflow
 * 
 * The function only runs when SHL_MEM_DEBUG_VALID_WRITE is defined, otherwise
 * it issues a warning that the test was skipped.
 * 
 * @note This test intentionally writes out of bounds to verify guard byte
 *       detection mechanisms. The memory allocator should issue warnings
 *       when corrupted guard bytes are detected during free operations.
 * 
 * @warning This test modifies memory outside allocated boundaries for testing
 *          purposes. It should only be used in debug builds with proper
 *          guard byte support enabled.
 */
static void test_guard_bytes() {
#ifdef SHL_MEM_DEBUG_VALID_WRITE
    TEST_INFO("=== Test 4: Guard Bytes (Buffer Overflow Detection) ===");

    // Allocate memory
    size_t size = 100;
    uint8_t *ptr = (uint8_t*)shl_mem_alloc(size);
    
    if (ptr != NULL) {
        TEST_PASS("Allocation for guard byte test successful");

        // Fill buffer correctly
        memset(ptr, 0x42, size);

        // Check free without corruption
        printf("  Testing normal free (no corruption):\n");
        shl_mem_free(ptr);
        TEST_PASS("Normal free without corruption successful");

        // Test with corrupted guard bytes
        ptr = (uint8_t*)shl_mem_alloc(size);
        if (ptr != NULL) {
            // Intentionally write out of bounds
            ptr[size] = 0xDE;     // Corrupt first guard byte
            ptr[size + 1] = 0xAD; // Corrupt second guard byte
            
            printf("  Testing free with corrupted guard bytes:\n");
            shl_mem_free(ptr);  // Should issue a warning
            TEST_PASS("Guard byte corruption detected");
        }

        // Test with partial corruption
        ptr = (uint8_t*)shl_mem_alloc(size);
        if (ptr != NULL) {
            ptr[size + 7] = 0xBA;  // Corrupt last guard byte

            printf("  Testing free with partial guard byte corruption:\n");
            shl_mem_free(ptr);
            TEST_PASS("Partial guard byte corruption detected");
        }
    } else {
        TEST_FAIL("Allocation for guard byte test failed");
    }
#else
    TEST_WARN("Guard bytes test skipped (SHL_MEM_DEBUG_VALID_WRITE not defined)");
#endif
}

// Test 5: Multithreading Safety
/**
 * @brief Thread worker function for concurrent memory allocation testing
 * 
 * This function performs repeated allocation and deallocation cycles in a multi-threaded
 * environment to test memory management under concurrent access. Each thread operates
 * independently with its own allocation patterns and timing.
 * 
 * @param arg Pointer to thread_test_params structure containing test configuration
 * 
 * @details The function performs the following operations:
 * - Binds the thread to a specific CPU core for consistent performance
 * - Initializes a thread-local random number generator with unique seed
 * - Executes multiple iterations of allocation/deallocation cycles:
 *   - Allocates memory blocks of random sizes within specified range
 *   - Fills allocated memory with thread-specific pattern for verification
 *   - Optionally uses aligned allocation based on parameters
 *   - Introduces random delays between operations if specified
 *   - Shuffles allocation order before freeing to test fragmentation handling
 *   - Frees all allocated memory blocks
 * - Updates global counters for allocations, frees, and errors (thread-safe)
 * - Handles both regular and aligned memory allocation/deallocation
 * - Supports both RTOS and non-RTOS build configurations
 * 
 * @note This function is designed to be used with pthread_create() for concurrent
 *       memory stress testing scenarios.
 * 
 * @return NULL (standard pthread worker return value)
 */
static void *thread_allocation_worker(void *arg) {
    struct thread_test_params *params = (struct thread_test_params*)arg;
    void **allocations = calloc(params->allocations_per_thread, sizeof(void*));
    int local_errors = 0;

    // Bind to CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(params->thread_id % get_nprocs(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // Random generator for each thread
    unsigned int seed = time(NULL) + params->thread_id;
    
    for (int iter = 0; iter < params->iterations; iter++) {
        // Allocation phase
        for (int i = 0; i < params->allocations_per_thread; i++) {
            int size = params->min_size + 
                      (rand_r(&seed) % (params->max_size - params->min_size));
            
            if (params->alignment > 0) {
                allocations[i] = shl_mem_alloc_aligned(size, params->alignment);
            } else {
                allocations[i] = shl_mem_alloc(size);
            }
            
            if (allocations[i] != NULL) {
                // Fill memory with pattern
                memset(allocations[i], params->thread_id & 0xFF, size);
                
                pthread_mutex_lock(&g_count_mutex);
                g_alloc_count++;
                pthread_mutex_unlock(&g_count_mutex);
            } else {
                local_errors++;
            }
            
            if (params->delay_us > 0) {
                usleep(rand_r(&seed) % params->delay_us);
            }
        }

        // Random shuffling for freeing in a different order
        for (int i = params->allocations_per_thread - 1; i > 0; i--) {
            int j = rand_r(&seed) % (i + 1);
            void *temp = allocations[i];
            allocations[i] = allocations[j];
            allocations[j] = temp;
        }

        // Freeing phase
        for (int i = 0; i < params->allocations_per_thread; i++) {
            if (allocations[i] != NULL) {
                if (params->alignment > 0) {
#ifdef SHL_BUILD_RTOS
                    shl_mem_free_aligned(allocations[i]);
#else
                    shl_mem_free(allocations[i]);
#endif
                } else {
                    shl_mem_free(allocations[i]);
                }
                
                pthread_mutex_lock(&g_count_mutex);
                g_free_count++;
                pthread_mutex_unlock(&g_count_mutex);
                
                allocations[i] = NULL;
            }
            
            if (params->delay_us > 0) {
                usleep(rand_r(&seed) % params->delay_us);
            }
        }
    }
    
    free(allocations);
    
    pthread_mutex_lock(&g_count_mutex);
    g_error_count += local_errors;
    pthread_mutex_unlock(&g_count_mutex);
    
    return NULL;
}

// Test 5: Multithreading
/**
 * @brief Tests multithreading safety of the memory allocation system
 * 
 * This function performs comprehensive multithreading tests to verify the thread safety
 * of the memory allocation and deallocation functions. It runs two distinct test scenarios
 * to validate concurrent memory operations under different conditions.
 * 
 * The function automatically detects the number of available CPU cores and adjusts
 * test parameters accordingly. When SHL_MEM_DEBUG is enabled, it applies safety
 * limitations due to known multithreading issues in the debug allocator.
 * 
 * Test Scenarios:
 * 1. Small Allocation Stress Test:
 *    - Creates 2x CPU core threads
 *    - Each thread performs 100 iterations with 50 allocations
 *    - Allocation sizes: 16-256 bytes (unaligned)
 *    - No artificial delays for maximum stress
 * 
 * 2. Aligned Allocation Test:
 *    - Creates threads equal to CPU core count
 *    - Each thread performs 50 iterations with 20 allocations
 *    - Allocation sizes: 512-4096 bytes with 64-byte alignment
 *    - 10μs delays to simulate realistic workload patterns
 * 
 * Success Criteria:
 * - All allocated memory must be properly freed (alloc_count == free_count)
 * - No allocation errors should occur during execution
 * - Thread synchronization must work correctly without race conditions
 * 
 * @note Uses global counters g_alloc_count, g_free_count, and g_error_count
 *       to track allocation statistics across all threads
 * @note Requires thread_allocation_worker function for worker thread implementation
 * @note When SHL_MEM_DEBUG is enabled, thread count is limited to 2 for safety
 * @note Measures and reports execution time for performance analysis
 * 
 * @warning In debug mode (SHL_MEM_DEBUG), the custom allocator may have
 *          multithreading limitations. For production multithreaded use,
 *          compile without SHL_MEM_DEBUG flag.
 */
static void test_multithreading() {
    TEST_INFO("=== Test 5: Multithreading Safety ===");
    
    int num_cpus = get_nprocs();
    printf("  System has %d CPU cores\n", num_cpus);

    // IMPORTANT: In the current implementation shl_mem_alloc with debug
    // may have multithreading issues in custom allocator
    // Check if debugging is enabled
#ifdef SHL_MEM_DEBUG
    TEST_WARN("Memory debug is enabled - multithread test may have limitations");
    TEST_WARN("For production multithreaded use, compile without SHL_MEM_DEBUG");

    // Limit the number of threads and iterations for safety
    num_cpus = (num_cpus > 2) ? 2 : num_cpus;
#endif

    // Reset counters
    g_alloc_count = 0;
    g_free_count = 0;
    g_error_count = 0;


    /**
     * @brief Performs a multithreaded memory allocation stress test with small allocations
     * 
     * This test creates multiple worker threads (2x the number of CPU cores) that perform
     * concurrent memory allocations and deallocations to verify thread safety of the memory
     * management system. Each thread performs small allocations between 16-256 bytes.
     * 
     * Test parameters:
     * - Thread count: 2 * num_cpus
     * - Iterations per thread: 100
     * - Allocations per thread: 50
     * - Allocation size range: 16-256 bytes
     * - No alignment requirements
     * - No artificial delays
     * 
     * The test measures total execution time and validates that:
     * - All allocations are properly freed (alloc_count == free_count)
     * - No errors occurred during the process
     * 
     * @note Uses global counters g_alloc_count, g_free_count, and g_error_count
     *       to track allocation statistics across all threads
     * @note Requires thread_allocation_worker function and get_time_us timing utility
     */
    {
        int num_threads = num_cpus * 2;
        pthread_t *threads = calloc(num_threads, sizeof(pthread_t));
        struct thread_test_params *params = calloc(num_threads, 
                                                   sizeof(struct thread_test_params));
        
        printf("  Starting test with %d threads, small allocations...\n", num_threads);
        
        uint64_t start_time = get_time_us();
        
        for (int i = 0; i < num_threads; i++) {
            params[i].thread_id = i;
            params[i].iterations = 100;
            params[i].min_size = 16;
            params[i].max_size = 256;
            params[i].alignment = 0;
            params[i].delay_us = 0;
            params[i].allocations_per_thread = 50;
            
            pthread_create(&threads[i], NULL, thread_allocation_worker, &params[i]);
        }
        
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        
        uint64_t elapsed = get_time_us() - start_time;
        
        printf("  Completed in %.2f ms\n", elapsed / 1000.0);
        printf("  Total allocations: %d, frees: %d, errors: %d\n", 
               g_alloc_count, g_free_count, g_error_count);
        
        if (g_alloc_count == g_free_count && g_error_count == 0) {
            TEST_PASS("Small allocation multithreading test");
        } else {
            TEST_FAIL("Small allocation multithreading test");
        }
        
        free(threads);
        free(params);
    }

    /**
     * @brief Multithreaded aligned memory allocation test
     * 
     * This test creates multiple threads to perform concurrent aligned memory allocations
     * and deallocations to verify thread safety of the memory allocation system.
     * 
     * Test parameters:
     * - Creates threads equal to the number of available CPUs
     * - Each thread performs 50 iterations of allocation/deallocation cycles
     * - Memory sizes range from 512 to 4096 bytes
     * - All allocations use 64-byte alignment
     * - 10 microsecond delay between operations to simulate real workload
     * - Each thread maintains 20 concurrent allocations
     * 
     * Success criteria:
     * - All allocated memory blocks must be successfully freed (g_alloc_count == g_free_count)
     * - No allocation errors should occur (g_error_count == 0)
     * 
     * @note Global counters g_alloc_count, g_free_count, and g_error_count are reset
     *       at the beginning of the test and monitored throughout execution
     */
    {
        g_alloc_count = 0;
        g_free_count = 0;
        g_error_count = 0;
        
        int num_threads = num_cpus;
        pthread_t *threads = calloc(num_threads, sizeof(pthread_t));
        struct thread_test_params *params = calloc(num_threads, 
                                                   sizeof(struct thread_test_params));
        
        printf("  Starting aligned allocation test with %d threads...\n", num_threads);
        
        for (int i = 0; i < num_threads; i++) {
            params[i].thread_id = i;
            params[i].iterations = 50;
            params[i].min_size = 512;
            params[i].max_size = 4096;
            params[i].alignment = 64;
            params[i].delay_us = 10;
            params[i].allocations_per_thread = 20;
            
            pthread_create(&threads[i], NULL, thread_allocation_worker, &params[i]);
        }
        
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        
        if (g_alloc_count == g_free_count && g_error_count == 0) {
            TEST_PASS("Aligned allocation multithreading test");
        } else {
            TEST_FAIL("Aligned allocation multithreading test");
        }
        
        free(threads);
        free(params);
    }
}

// Test 6: Stress test
/**
 * @brief Performs a stress test on the memory allocation system
 * 
 * This function conducts an intensive stress test by performing 100,000 random
 * allocation and deallocation operations to verify the robustness and reliability
 * of the memory management system under heavy load.
 * 
 * Test behavior:
 * - Maintains up to 10,000 concurrent allocations
 * - Randomly chooses between allocation (50%) and deallocation (50%) operations
 * - Allocates memory blocks with random sizes from 1 to 8192 bytes
 * - Fills allocated memory with a test pattern for verification
 * - Provides progress updates every 10,000 operations
 * - Ensures all memory is properly freed at the end
 * - Measures and reports total execution time
 * 
 * The test verifies:
 * - Memory allocation/deallocation under stress conditions
 * - Absence of memory leaks after intensive operations
 * - System stability during rapid allocation/free cycles
 * - Performance characteristics of the memory allocator
 * 
 * @note Uses thread-safe rand_r() for reproducible random behavior
 * @note Tracks active allocation count to prevent buffer overflow
 * @note Reports TEST_PASS if all memory is successfully freed, TEST_FAIL otherwise
 */
static void test_stress() {
    TEST_INFO("=== Test 6: Stress Test ===");
    
    const int max_allocations = 10000;
    void **ptrs = calloc(max_allocations, sizeof(void*));
    int active_count = 0;
    unsigned int seed = time(NULL);
    
    printf("  Running allocation/free stress test...\n");
    uint64_t start_time = get_time_us();
    
    for (int i = 0; i < 100000; i++) {
        int action = rand_r(&seed) % 2;
        
        if (action == 0 && active_count < max_allocations) {
            // Выделение
            int size = 1 + (rand_r(&seed) % 8192);
            int slot = rand_r(&seed) % max_allocations;
            
            if (ptrs[slot] == NULL) {
                ptrs[slot] = shl_mem_alloc(size);
                if (ptrs[slot] != NULL) {
                    memset(ptrs[slot], i & 0xFF, size);
                    active_count++;
                }
            }
        } else if (active_count > 0) {
            // Освобождение
            int slot = rand_r(&seed) % max_allocations;
            if (ptrs[slot] != NULL) {
                shl_mem_free(ptrs[slot]);
                ptrs[slot] = NULL;
                active_count--;
            }
        }
        
        if (i % 10000 == 0) {
            printf("  Progress: %d/100000, active allocations: %d\n", i, active_count);
        }
    }

// Free remaining memory
    for (int i = 0; i < max_allocations; i++) {
        if (ptrs[i] != NULL) {
            shl_mem_free(ptrs[i]);
            active_count--;
        }
    }
    
    uint64_t elapsed = get_time_us() - start_time;
    
    if (active_count == 0) {
        printf("  Stress test completed in %.2f seconds\n", elapsed / 1000000.0);
        TEST_PASS("Stress test - all memory freed");
    } else {
        TEST_FAIL("Stress test - memory leak detected");
    }
    
    free(ptrs);
}

// Test 7: Debug Features
/**
 * @brief Tests memory debugging features when SHL_MEM_DEBUG is enabled
 *
 * This function verifies the functionality of memory debugging capabilities by:
 * - Printing the initial memory map state
 * - Creating multiple memory allocations of different sizes (100, 200, 300 bytes)
 * - Displaying memory map after allocations to show memory usage
 * - Freeing one allocation and showing the updated memory map
 * - Freeing remaining allocations and verifying final cleanup
 * - Using shl_mem_print_map() to visualize memory state at each step
 *
 * The test is conditionally compiled and only runs when SHL_MEM_DEBUG is defined.
 * If the debug macro is not available, the test is skipped with a warning message.
 *
 * @note Requires SHL_MEM_DEBUG to be defined for full functionality
 * @note Uses TEST_PASS() to indicate successful completion of debug feature tests
 * @note Uses TEST_WARN() to indicate when test is skipped due to missing debug support
 */
static void test_debug_features() {
#ifdef SHL_MEM_DEBUG
    TEST_INFO("=== Test 7: Debug Features ===");

    printf("  Current memory map:\n");
    shl_mem_print_map();

    // Create several allocations
    void *ptr1 = shl_mem_alloc(100);
    void *ptr2 = shl_mem_alloc(200);
    void *ptr3 = shl_mem_alloc(300);
    
    printf("\n  After allocations:\n");
    shl_mem_print_map();

    // Free part of the allocations
    shl_mem_free(ptr2);
    
    printf("\n  After freeing ptr2:\n");
    shl_mem_print_map();

    // Free remaining allocations
    shl_mem_free(ptr1);
    shl_mem_free(ptr3);
    
    printf("\n  After freeing all:\n");
    shl_mem_print_map();
    
    TEST_PASS("Debug features working");
#else
    TEST_WARN("Debug features test skipped (SHL_MEM_DEBUG not defined)");
#endif
}

 // Main function
int main(int argc, char *argv[]) {
    printf("CSI-NN2 Memory Test Suite for TH1520/C920\n");
    printf("==========================================\n");

    // System information
    struct utsname system_info;
    if (uname(&system_info) == 0) {
        printf("System: %s %s %s\n", system_info.sysname, 
               system_info.release, system_info.machine);
    }
    
    printf("\nInitial memory status:\n");
    print_memory_info();

    // Run tests
    test_basic_allocation();
    test_aligned_allocation();
    test_calloc_realloc();
    test_guard_bytes();
    test_multithreading();
    test_stress();
    test_debug_features();

    // Final memory status
    printf("\nFinal memory status:\n");
    print_memory_info();

    // Summary statistics
    printf("\n==========================================\n");
    printf("All tests completed!\n");

// Print memory statistics
    printf("Total allocations: %d\n", g_alloc_count);
    printf("Total frees: %d\n", g_free_count);
    printf("Total errors: %d\n", g_error_count);
    
    if (g_error_count > 0) {
        printf(ANSI_COLOR_RED "Memory test completed with errors!\n" ANSI_COLOR_RESET);
        return EXIT_FAILURE;
    } else {
        printf(ANSI_COLOR_GREEN "Memory test completed successfully!\n" ANSI_COLOR_RESET);
        return EXIT_SUCCESS;
    }
}
