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
#ifndef INCLUDE_SHL_MEMORY_H_
#define INCLUDE_SHL_MEMORY_H_

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Print memory allocation debug map
 * 
 * Only available when SHL_MEM_DEBUG is defined
 */
void shl_mem_print_map();

/**
 * @brief Allocate memory
 * 
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory or NULL on failure
 * 
 * @note Returns NULL for size <= 0
 * @note Thread-safe when SHL_MEM_DEBUG is defined
 */
void *shl_mem_alloc(int64_t size);

/**
 * @brief Allocate aligned memory
 * 
 * @param size Size in bytes to allocate
 * @param aligned_bytes Alignment requirement (must be power of 2)
 * @return Pointer to aligned memory or NULL on failure
 * 
 * @note For RTOS builds, use shl_mem_free_aligned() to free
 * @note For non-RTOS builds, use shl_mem_free() to free
 */
void *shl_mem_alloc_aligned(int64_t size, int aligned_bytes);

/**
 * @brief Allocate and zero memory
 * 
 * @param nmemb Number of elements
 * @param size Size of each element
 * @return Pointer to allocated memory or NULL on failure
 * 
 * @note Checks for multiplication overflow
 */
void *shl_mem_calloc(size_t nmemb, size_t size);

/**
 * @brief Reallocate memory
 * 
 * @param ptr Pointer to existing memory (can be NULL)
 * @param size New size
 * @param orig_size Original size (0 if unknown)
 * @return Pointer to reallocated memory or NULL on failure
 * 
 * @note If orig_size is 0, only 'size' bytes are copied
 * @note Original memory is freed on success
 */
void *shl_mem_realloc(void *ptr, size_t size, size_t orig_size);

/**
 * @brief Free memory
 * 
 * @param ptr Pointer to memory to free (can be NULL)
 */
void shl_mem_free(void *ptr);

#ifdef SHL_BUILD_RTOS
/**
 * @brief Free aligned memory (RTOS only)
 * 
 * @param ptr Pointer to aligned memory to free
 * 
 * @note Only use for memory allocated with shl_mem_alloc_aligned()
 */
void shl_mem_free_aligned(void *ptr);
#endif

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SHL_MEMORY_H_
