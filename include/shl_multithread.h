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
#ifndef INCLUDE_SHL_MULTITHREAD_H_
#define INCLUDE_SHL_MULTITHREAD_H_

#if (!defined SHL_BUILD_RTOS)
#include <omp.h>
#endif
#include "csinn/csi_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup multithread Multi-threading Support Functions
 * @{
 */

/**
 * @brief Set the number of threads for parallel execution
 * 
 * @param threads Number of threads to use (must be >= 1)
 * 
 * @note If OpenMP is not available, this function will issue a warning
 * @note If threads exceeds available processors, a debug message will be printed
 * 
 * @thread_safety This function is thread-safe. Multiple threads can call it
 *                simultaneously, but the behavior is serialized internally.
 *                It's recommended to call this function from a single thread
 *                before starting parallel computations.
 */
void shl_multithread_set_threads(int threads);

/**
 * @brief Get the current number of threads configured
 * 
 * @return Current thread count (always returns 1 if OpenMP is not available)
 * 
 * @thread_safety This function is thread-safe and can be called from any thread.
 */
int shl_multithread_get_threads();

/**
 * @brief Check if multithreading is enabled
 * 
 * @return CSINN_TRUE if more than 1 thread is configured, CSINN_FALSE otherwise
 * 
 * @thread_safety This function is thread-safe and can be called from any thread.
 */
int shl_multithread_is_enable();

/**
 * @brief Get the maximum number of threads available on the system
 * 
 * @return Maximum thread count (number of processors)
 * 
 * @thread_safety This function is thread-safe and can be called from any thread.
 */
int shl_multithread_get_max_threads();

/**
 * @brief Synchronization barrier for all threads
 * 
 * @note This function only has effect when called from within an OpenMP
 *       parallel region. Outside of parallel regions, it's a no-op.
 * 
 * @thread_safety This function must only be called from within OpenMP parallel
 *                regions where it's designed to synchronize threads.
 */
void shl_multithread_sync();

/** @} */ // end of multithread group

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_MULTITHREAD_H_
