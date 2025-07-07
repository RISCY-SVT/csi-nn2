/**
 * @file test_memory_multithread.c
 * @brief Multithreaded stress test for CSI-NN2 memory allocation system
 * 
 * This program performs comprehensive stress testing of the shl_memory allocation
 * system under multithreaded conditions to detect race conditions, memory leaks,
 * and corruption issues. The test creates multiple worker threads that perform
 * concurrent memory allocations and deallocations with pattern verification.
 * 
 * @details Test Features:
 * - Creates 2x CPU core count threads for maximum stress testing
 * - Each thread performs 1000 iterations of allocation/deallocation cycles
 * - Supports up to 10,000 concurrent allocations per thread
 * - Random allocation sizes between 16 bytes and 8KB
 * - 70% allocation probability, 30% deallocation probability per cycle
 * - Memory corruption detection through pattern verification
 * - Thread affinity binding to specific CPU cores
 * - Comprehensive statistics collection and reporting
 * 
 * @note Requires _GNU_SOURCE for pthread affinity functions
 * @note Uses thread-safe rand_r() for random number generation
 * @note Includes periodic delays to increase race condition likelihood
 * 
 * @return 0 if all tests pass (no leaks, no errors), 1 if failures detected
 * 
 * @author Custler
 * @version 1.1
 * @date 2025
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>       // For usleep()
#include <sys/sysinfo.h>  // For get_nprocs()
#include <time.h>         // For time()
#include <sched.h>        // For CPU_SET, CPU_ZERO

// Do not enable debug features for a clean multithreading test
#include "shl_memory.h"

#define NUM_ITERATIONS 1000
#define NUM_ALLOCATIONS 10000
#define MIN_SIZE 16
#define MAX_SIZE 8192

struct thread_data {
    int thread_id;
    int success_count;
    int error_count;
    uint64_t total_allocated;
    uint64_t total_freed;
};

/**
 * @brief Worker function executed by each thread in the multithreaded memory test
 * 
 * This function performs stress testing of the memory allocation system by:
 * - Setting thread affinity to a specific CPU core to reduce contention
 * - Performing multiple iterations of random memory allocations and deallocations
 * - Filling allocated memory with thread-specific patterns for corruption detection
 * - Verifying memory integrity before deallocation to detect corruption
 * - Collecting statistics on allocation success, failures, and total memory usage
 * 
 * The function simulates realistic memory usage patterns with a 70% allocation
 * probability and 30% deallocation probability per iteration. Memory corruption
 * is detected by verifying that allocated memory still contains the expected
 * pattern before freeing.
 * 
 * @param arg Pointer to thread_data structure containing thread configuration
 *            and statistics collection variables
 * @return NULL (unused return value for pthread compatibility)
 * 
 * @note Uses rand_r() for thread-safe random number generation
 * @note Includes periodic delays to increase likelihood of race condition detection
 * @note Ensures all allocated memory is freed before thread termination
 */
void *thread_worker(void *arg) {
    struct thread_data *data = (struct thread_data *)arg;
    void *ptrs[NUM_ALLOCATIONS];
    size_t sizes[NUM_ALLOCATIONS];
    unsigned int seed = time(NULL) + data->thread_id;
    
    // Set thread affinity to a specific CPU core
    // This helps to reduce contention and improve performance
    // by binding each thread to a specific core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(data->thread_id % get_nprocs(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    memset(ptrs, 0, sizeof(ptrs));
    memset(sizes, 0, sizeof(sizes));
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Random allocations 
        // Try to allocate memory in a round-robin fashion
        // with a 70% chance of success
        // and a 30% chance of deallocation
        for (int i = 0; i < NUM_ALLOCATIONS; i++) {
            if (ptrs[i] == NULL && (rand_r(&seed) % 100) < 70) {
                sizes[i] = MIN_SIZE + (rand_r(&seed) % (MAX_SIZE - MIN_SIZE));
                ptrs[i] = shl_mem_alloc(sizes[i]);
                
                if (ptrs[i] != NULL) {
                    // Fill memory with pattern
                    memset(ptrs[i], (data->thread_id + i) & 0xFF, sizes[i]);
                    data->total_allocated += sizes[i];
                    data->success_count++;
                } else {
                    data->error_count++;
                }
            }
        }
        
        // Random deallocations
        // Free memory with a 30% chance
        // and check for memory corruption
        // by verifying the pattern before freeing
        // This simulates a realistic scenario where threads
        // may randomly allocate and free memory
        // while also checking for potential memory corruption
        for (int i = 0; i < NUM_ALLOCATIONS; i++) {
            if (ptrs[i] != NULL && (rand_r(&seed) % 100) < 30) {
                // Check pattern before freeing
                uint8_t expected = (data->thread_id + i) & 0xFF;
                uint8_t *bytes = (uint8_t*)ptrs[i];
                int corrupt = 0;
                
                for (size_t j = 0; j < sizes[i]; j++) {
                    if (bytes[j] != expected) {
                        corrupt = 1;
                        break;
                    }
                }
                
                if (corrupt) {
                    printf("Thread %d: Memory corruption detected!\n", data->thread_id);
                    data->error_count++;
                }
                
                shl_mem_free(ptrs[i]);
                data->total_freed += sizes[i];
                ptrs[i] = NULL;
                sizes[i] = 0;
            }
        }
        
        // A small delay to increase the likelihood of race conditions
        if (iter % 100 == 0) {
            usleep(1);
        }
    }
    
    // Free remaining memory
    // at the end of the thread's execution
    // This ensures that any memory allocated during the test
    // is properly released before the thread exits
    for (int i = 0; i < NUM_ALLOCATIONS; i++) {
        if (ptrs[i] != NULL) {
            shl_mem_free(ptrs[i]);
            data->total_freed += sizes[i];
        }
    }
    
    return NULL;
}

int main() {
    printf("CSI-NN2 Multithread Memory Test (No Debug)\n");
    printf("==========================================\n");
    
    int num_threads = get_nprocs() * 2;  // Double the number of CPUs for load
    printf("Running with %d threads on %d CPUs\n", num_threads, get_nprocs());
    
    pthread_t *threads = calloc(num_threads, sizeof(pthread_t));
    struct thread_data *thread_data = calloc(num_threads, sizeof(struct thread_data));
    
    if (threads == NULL || thread_data == NULL) {
        printf("Failed to allocate memory for test infrastructure\n");
        free(threads);
        free(thread_data);
        return 1;
    }
    
    // Start threads
    uint64_t start_time = time(NULL);
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        if (pthread_create(&threads[i], NULL, thread_worker, &thread_data[i]) != 0) {
            printf("Failed to create thread %d\n", i);
            // Clean up already created threads
            for (int j = 0; j < i; j++) {
                pthread_cancel(threads[j]);
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(thread_data);
            return 1;
        }
    }
    
    // Wait for threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    uint64_t elapsed = time(NULL) - start_time;

    // Statistics
    uint64_t total_allocated = 0;
    uint64_t total_freed = 0;
    int total_success = 0;
    int total_errors = 0;
    
    printf("\nPer-thread statistics:\n");
    for (int i = 0; i < num_threads; i++) {
        printf("Thread %2d: %d allocations, %d errors, %lu allocated, %lu freed\n",
               i, thread_data[i].success_count, thread_data[i].error_count,
               thread_data[i].total_allocated, thread_data[i].total_freed);
        
        total_allocated += thread_data[i].total_allocated;
        total_freed += thread_data[i].total_freed;
        total_success += thread_data[i].success_count;
        total_errors += thread_data[i].error_count;
    }
    
    printf("\nTotal statistics:\n");
    printf("  Successful allocations: %d\n", total_success);
    printf("  Errors: %d\n", total_errors);
    printf("  Total allocated: %lu bytes\n", total_allocated);
    printf("  Total freed: %lu bytes\n", total_freed);
    printf("  Memory leak: %ld bytes\n", (long)(total_allocated - total_freed));
    printf("  Time elapsed: %lu seconds\n", elapsed);
    
    // Determine test result
    int test_result = 0;
    if (total_allocated == total_freed && total_errors == 0) {
        printf("\nTEST PASSED: All memory properly managed, no errors\n");
        test_result = 0;
    } else {
        printf("\nTEST FAILED: Memory leaks or errors detected\n");
        test_result = 1;
    }
    
    // Clean up test infrastructure
    free(threads);
    free(thread_data);
    
    return test_result;
}
