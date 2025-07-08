/*
 * Copyright (C) 2016-2023 C-SKY Microsystems Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "shl_debug.h"
#include "shl_multithread.h"
#include <stdlib.h>
#include <stdatomic.h>

#ifndef SHL_BUILD_RTOS
#include <pthread.h>
#endif

// Always use atomic for thread count to avoid data races
static _Atomic int shl_thread_num = ATOMIC_VAR_INIT(1);
static _Atomic int shl_thread_initialized = ATOMIC_VAR_INIT(0);

#ifdef _OPENMP
// Mutex for protecting OpenMP calls
#ifndef SHL_BUILD_RTOS
static pthread_mutex_t omp_mutex = PTHREAD_MUTEX_INITIALIZER;
#else
// For RTOS, define appropriate mutex type
static omp_lock_t omp_mutex;
static _Atomic int omp_mutex_initialized = ATOMIC_VAR_INIT(0);

static void ensure_omp_mutex_init(void)
{
    if (atomic_exchange(&omp_mutex_initialized, 1) == 0) {
        omp_init_lock(&omp_mutex);
    }
}
#endif

// Lazy initialization using pthread_once or atomic flag
#ifndef SHL_BUILD_RTOS
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
#endif

static void shl_multithread_init_internal(void)
{
    const char* env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads) {
        int threads = atoi(env_threads);
        if (threads > 0) {
            atomic_store_explicit(&shl_thread_num, threads, memory_order_relaxed);
#ifndef SHL_BUILD_RTOS
            pthread_mutex_lock(&omp_mutex);
#else
            omp_set_lock(&omp_mutex);
#endif
            omp_set_num_threads(threads);
            omp_set_dynamic(0);  // Disable dynamic adjustment
#ifndef SHL_BUILD_RTOS
            pthread_mutex_unlock(&omp_mutex);
#else
            omp_unset_lock(&omp_mutex);
#endif
        }
    } else {
        // If OMP_NUM_THREADS not set, use all available processors
        int num_procs = omp_get_num_procs();
        if (num_procs > 1) {
            atomic_store_explicit(&shl_thread_num, num_procs, memory_order_relaxed);
#ifndef SHL_BUILD_RTOS
            pthread_mutex_lock(&omp_mutex);
#else
            omp_set_lock(&omp_mutex);
#endif
            omp_set_num_threads(num_procs);
            omp_set_dynamic(0);  // Disable dynamic adjustment
#ifndef SHL_BUILD_RTOS
            pthread_mutex_unlock(&omp_mutex);
#else
            omp_unset_lock(&omp_mutex);
#endif
        }
    }
}

static void ensure_initialized(void)
{
#ifndef SHL_BUILD_RTOS
    pthread_once(&init_once, shl_multithread_init_internal);
#else
    if (atomic_load_explicit(&shl_thread_initialized, memory_order_acquire) == 0) {
        if (atomic_exchange(&shl_thread_initialized, 1) == 0) {
            ensure_omp_mutex_init();
            shl_multithread_init_internal();
        }
    }
#endif
}
#endif  // _OPENMP

void shl_multithread_set_threads(int threads)
{
#ifdef _OPENMP
    ensure_initialized();
    
    // Validate input
    if (threads < 1) {
        shl_debug_warning("Invalid thread count: %d, using 1 instead\n", threads);
        threads = 1;
    }
    
    // Optional: warn if exceeding available processors
    int max_threads = omp_get_num_procs();
    if (threads > max_threads) {
        shl_debug_info("Thread count %d exceeds available processors %d\n", 
                       threads, max_threads);
    }
    
    // Thread-safe update of both our counter and OpenMP
#ifndef SHL_BUILD_RTOS
    pthread_mutex_lock(&omp_mutex);
#else
    ensure_omp_mutex_init();
    omp_set_lock(&omp_mutex);
#endif
    
    omp_set_num_threads(threads);
    omp_set_dynamic(0);  // Ensure OpenMP respects our setting
    atomic_store_explicit(&shl_thread_num, threads, memory_order_release);
    
#ifndef SHL_BUILD_RTOS
    pthread_mutex_unlock(&omp_mutex);
#else
    omp_unset_lock(&omp_mutex);
#endif
#else
    if (threads != 1) {
        shl_debug_warning("OPENMP is not defined! Thread count will remain 1\n");
    }
    atomic_store_explicit(&shl_thread_num, 1, memory_order_relaxed);
#endif
}

int shl_multithread_get_threads()
{
#ifdef _OPENMP
    ensure_initialized();
#endif
    return atomic_load_explicit(&shl_thread_num, memory_order_relaxed);
}

int shl_multithread_is_enable()
{
#ifdef _OPENMP
    ensure_initialized();
    // Simply check if we have more than 1 thread configured
    // No need to call omp_set_num_threads here
    return (atomic_load_explicit(&shl_thread_num, memory_order_relaxed) > 1) ? CSINN_TRUE : CSINN_FALSE;
#else
    return CSINN_FALSE;
#endif
}

int shl_multithread_get_max_threads()
{
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}

void shl_multithread_sync()
{
#ifdef _OPENMP
    // Only valid inside parallel region
    if (omp_in_parallel()) {
        #pragma omp barrier
    }
    // Outside parallel region, threads are already synchronized
#endif
}
