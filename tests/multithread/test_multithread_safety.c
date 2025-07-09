#include <pthread.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "shl_multithread.h"

#define NUM_THREADS 10
#define NUM_ITERATIONS 1000

// Global atomic flag for test failures
static atomic_bool test_failed = ATOMIC_VAR_INIT(false);

typedef struct {
    int thread_id;
    int *results;
} thread_data_t;

// Thread that constantly changes thread count
void* thread_setter(void* arg) {
    thread_data_t *data = (thread_data_t*)arg;
    int thread_id = data->thread_id;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        int num_threads = (thread_id % 4) + 1;
        shl_multithread_set_threads(num_threads);
        
        // Small random delay to increase chance of race conditions
        usleep(rand() % 100);
    }
    return NULL;
}

// Thread that constantly reads thread status
void* thread_checker(void* arg) {
    thread_data_t *data = (thread_data_t*)arg;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // Check if multithreading is enabled
        int enabled = shl_multithread_is_enable();
        if (enabled != CSINN_TRUE && enabled != CSINN_FALSE) {
            fprintf(stderr, "ERROR: Invalid return value from shl_multithread_is_enable(): %d\n", enabled);
            atomic_store(&test_failed, true);
        }
        
        // Get current thread count
        int thread_count = shl_multithread_get_threads();
        if (thread_count < 1) {
            fprintf(stderr, "ERROR: Invalid thread count: %d\n", thread_count);
            atomic_store(&test_failed, true);
        }
        
        // Store result for validation
        data->results[i] = thread_count;
        
        // Small random delay
        usleep(rand() % 100);
    }
    return NULL;
}

// Thread that reads max threads
void* thread_max_reader(void* arg) {
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        int max_threads = shl_multithread_get_max_threads();
        if (max_threads < 1) {
            fprintf(stderr, "ERROR: Invalid max threads: %d\n", max_threads);
            atomic_store(&test_failed, true);
        }
        
        usleep(rand() % 100);
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    int *results[NUM_THREADS];
    
    // Initialize random number generator
    srand(time(NULL));
    
    printf("Starting multithread safety test...\n");
    printf("Number of processors: %d\n", shl_multithread_get_max_threads());
    
    // Allocate result arrays
    for (int i = 0; i < NUM_THREADS; i++) {
        results[i] = (int*)calloc(NUM_ITERATIONS, sizeof(int));
        thread_data[i].thread_id = i;
        thread_data[i].results = results[i];
    }
    
    // Test 1: Mixed readers and writers
    printf("\nTest 1: Mixed readers and writers\n");
    
    // Start setter threads (30% of threads)
    for (int i = 0; i < NUM_THREADS * 3 / 10; i++) {
        pthread_create(&threads[i], NULL, thread_setter, &thread_data[i]);
    }
    
    // Start checker threads (60% of threads)
    for (int i = NUM_THREADS * 3 / 10; i < NUM_THREADS * 9 / 10; i++) {
        pthread_create(&threads[i], NULL, thread_checker, &thread_data[i]);
    }
    
    // Start max reader threads (10% of threads)
    for (int i = NUM_THREADS * 9 / 10; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_max_reader, &thread_data[i]);
    }
    
    // Wait for all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    if (atomic_load(&test_failed)) {
        printf("Test 1 FAILED!\n");
        return 1;
    }
    printf("Test 1 passed!\n");
    
    // Test 2: Stress test with all writers
    printf("\nTest 2: Stress test with all writers\n");
    atomic_store(&test_failed, false);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, thread_setter, &thread_data[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    if (atomic_load(&test_failed)) {
        printf("Test 2 FAILED!\n");
        return 1;
    }
    printf("Test 2 passed!\n");
    
    // Test 3: Validate final state and consistency
    printf("\nTest 3: Validating final state and consistency\n");
    
    int final_threads = shl_multithread_get_threads();
    printf("Final thread count: %d\n", final_threads);
    if (final_threads < 1 || final_threads > 4) {
        printf("ERROR: Unexpected final thread count\n");
        return 1;
    }
    
    int is_enabled = shl_multithread_is_enable();
    if (final_threads > 1) {
        if (is_enabled != CSINN_TRUE) {
            printf("ERROR: Thread count > 1 but is_enable returns false\n");
            return 1;
        }
    } else {
        if (is_enabled != CSINN_FALSE) {
            printf("ERROR: Thread count = 1 but is_enable returns true\n");
            return 1;
        }
    }
    
    // Analyze result consistency
    printf("\nAnalyzing thread count consistency:\n");
    for (int i = NUM_THREADS * 3 / 10; i < NUM_THREADS * 9 / 10; i++) {
        int min_val = results[i][0];
        int max_val = results[i][0];
        for (int j = 1; j < NUM_ITERATIONS; j++) {
            if (results[i][j] < min_val) min_val = results[i][j];
            if (results[i][j] > max_val) max_val = results[i][j];
        }
        printf("Thread %d: min=%d, max=%d\n", i, min_val, max_val);
    }
    
    printf("Test 3 passed!\n");
    
    // Cleanup
    for (int i = 0; i < NUM_THREADS; i++) {
        free(results[i]);
    }
    
    printf("\nAll thread safety tests passed successfully!\n");
    return 0;
}
