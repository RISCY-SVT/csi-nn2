/**
 * @file memory.c
 * @brief Memory allocation and debugging utilities for SHL library
 * 
 * This module provides a comprehensive memory management system with optional
 * debugging capabilities, thread safety, and platform-specific optimizations.
 * It supports various build configurations including RTOS environments and
 * custom allocators.
 * 
 * @section features Features
 * - Safe memory allocation with overflow protection
 * - Optional memory leak tracking and debugging
 * - Buffer overrun detection with guard bytes
 * - Thread-safe operations using mutex protection
 * - Aligned memory allocation for performance-critical code
 * - Support for custom allocators (ATAT malloc)
 * - Cross-platform compatibility (POSIX and RTOS)
 * 
 * @section debug_modes Debug Modes
 * - SHL_MEM_DEBUG: Enables allocation tracking and leak detection
 * - SHL_MEM_DEBUG_VALID_WRITE: Adds guard bytes to detect buffer overruns
 * - SHL_USE_ATAT_MALLOC: Uses custom ATAT allocator instead of standard malloc
 * 
 * @section thread_safety Thread Safety
 * When SHL_MEM_DEBUG is enabled, all memory operations are protected by mutex
 * locks to ensure thread safety. For RTOS builds, mutex implementations need
 * to be provided by the platform.
 * 
 * @section alignment Memory Alignment
 * The module provides aligned memory allocation functions that ensure proper
 * memory alignment for performance-critical operations. The RTOS version
 * manually aligns memory, while POSIX systems use posix_memalign().
 * 
 * @note All allocation functions return NULL on failure and are designed to
 *       be safe replacements for standard C library memory functions.
 * 
 * @warning When using debug modes, ensure proper cleanup to avoid false
 *          memory leak reports. Guard byte corruption indicates potential
 *          buffer overflow bugs that should be investigated.
 */

#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>

#include "shl_utils.h"

// #define SHL_MEM_DEBUG
// #define SHL_MEM_DEBUG_VALID_WRITE
// #define SHL_USE_ATAT_MALLOC

#ifndef SHL_BUILD_RTOS
#include <pthread.h>
#endif

struct shl_mem_alloc_debug_element_ {
    void *ptr;
    int64_t size;
    int is_free;
};

struct shl_mem_alloc_debug_map_ {
    struct shl_mem_alloc_debug_element_ *element;
    int element_number;
    int index;
    int64_t total_size;
};

static struct shl_mem_alloc_debug_map_ shl_mem_alloc_debug_map;

// MUTEX_LOCK and MUTEX_UNLOCK are used to protect the memory debug map
#ifdef SHL_MEM_DEBUG
#ifndef SHL_BUILD_RTOS
static pthread_mutex_t shl_mem_debug_mutex = PTHREAD_MUTEX_INITIALIZER;
#define MUTEX_LOCK()   pthread_mutex_lock(&shl_mem_debug_mutex)
#define MUTEX_UNLOCK() pthread_mutex_unlock(&shl_mem_debug_mutex)
#else
// For RTOS, we need to define MUTEX_LOCK and MUTEX_UNLOCK
#define MUTEX_LOCK()   /* TODO: Implement for RTOS */
#define MUTEX_UNLOCK() /* TODO: Implement for RTOS */
#endif
#endif

void shl_mem_print_map()
{
#ifdef SHL_MEM_DEBUG
    MUTEX_LOCK();
    
    printf("total size = %ld\n", shl_mem_alloc_debug_map.total_size);
    for (int i = 0; i <= shl_mem_alloc_debug_map.index; i++) {
        struct shl_mem_alloc_debug_element_ *e = shl_mem_alloc_debug_map.element + i;
        printf("element %d: ptr = %p, size = %ld, is_free = %d\n", i, e->ptr, e->size, e->is_free);
    }
    
    MUTEX_UNLOCK();
#endif
}

/**
 * @brief Insert a memory allocation record into the debug tracking map
 * 
 * This function adds a new memory allocation entry to the global debug tracking
 * map. It automatically grows the tracking table when needed by allocating
 * additional slots in chunks of 512 elements.
 * 
 * @param ptr Pointer to the allocated memory block to track
 * @param size Size in bytes of the allocated memory block
 * 
 * @return 0 on success, -1 on failure
 * 
 * @retval 0 Successfully inserted the allocation record
 * @retval -1 Failed due to index overflow, table growth failure, or memory allocation error
 * 
 * @note This function MUST be called under mutex protection to ensure thread safety
 * @note Uses regular realloc() instead of shl_mem_alloc() to avoid recursion
 * @note Newly allocated table elements are zero-initialized
 * @note The tracking table grows in increments of 512 elements when full
 * 
 * @warning Not thread-safe - caller must ensure proper synchronization
 * @warning May fail if too many allocations are tracked (INT_MAX limit)
 */
static int shl_mem_map_insert(void *ptr, uint64_t size)
{
    // IMPORTANT: This function should only be called under mutex protection!

    int element_number = shl_mem_alloc_debug_map.element_number;
    int index = shl_mem_alloc_debug_map.index;
    
    // Check for index overflow
    if (index >= INT_MAX - 1) {
        shl_debug_error("Too many allocations tracked\n");
        return -1;
    }
    
    if (element_number == 0 || index >= element_number - 1) {
        // Check for overflow when increasing size
        if (element_number > INT_MAX - 512) {
            shl_debug_error("Cannot grow allocation tracking table\n");
            return -1;
        }
        
        int new_number = element_number + 512;
        size_t new_size = (size_t)new_number * sizeof(struct shl_mem_alloc_debug_element_);
        
        // Use regular realloc, as shl_mem_alloc may cause recursion
        struct shl_mem_alloc_debug_element_ *new_elements = realloc(
            shl_mem_alloc_debug_map.element, new_size);
        
        if (new_elements == NULL) {
            shl_debug_error("Failed to grow allocation tracking table\n");
            return -1;
        }
        
        // Initialize new elements to zero
        memset(new_elements + element_number, 0, 
               (new_number - element_number) * sizeof(struct shl_mem_alloc_debug_element_));
        
        shl_mem_alloc_debug_map.element_number = new_number;
        shl_mem_alloc_debug_map.element = new_elements;
    }
    
    shl_mem_alloc_debug_map.element[index].ptr = ptr;
    shl_mem_alloc_debug_map.element[index].size = size;
    shl_mem_alloc_debug_map.element[index].is_free = 0;
    shl_mem_alloc_debug_map.index++;
    return 0;
}

/**
 * @brief Allocate memory with error checking and debug features
 * 
 * This function provides a safe memory allocation wrapper with comprehensive
 * error checking, overflow protection, and optional debugging capabilities.
 * It is weakly defined to allow overriding in specific builds.
 * 
 * @param size The number of bytes to allocate (must be non-negative)
 * 
 * @return void* Pointer to allocated memory on success, NULL on failure
 * 
 * @details
 * - Validates input size for negative values and overflow conditions
 * - On 32-bit systems, checks if size exceeds SIZE_MAX
 * - When SHL_MEM_DEBUG_VALID_WRITE is enabled, adds 8-byte guard pattern
 *   at the end of allocation to detect buffer overruns
 * - When SHL_USE_ATAT_MALLOC is enabled, uses custom allocator
 * - When SHL_MEM_DEBUG is enabled, tracks allocations in debug map
 * - All allocated memory is zero-initialized via calloc
 * 
 * @note
 * - Returns NULL for zero-byte allocations with debug info
 * - Guard bytes contain magic pattern: 0xff, 0x23, 0x33, 0x44, 0x45, 0x55, 0x67, 0xff
 * - Thread-safe when SHL_MEM_DEBUG is enabled (uses mutex)
 * 
 * @warning
 * - Caller is responsible for freeing allocated memory
 * - Guard bytes should not be modified when debug mode is active
 */
__attribute__((weak)) void *shl_mem_alloc(int64_t size)
{
    void *ret;
    
    // Check for negative size
    if (size < 0) {
        shl_debug_error("Negative size requested: %lld\n", (long long)size);
        return NULL;
    }
    
    if (size == 0) {
        shl_debug_info("alloc 0 byte\n");
        return NULL;
    }
    
    // Check for overflow for 32-bit systems
    if (sizeof(size_t) < sizeof(int64_t) && size > (int64_t)SIZE_MAX) {
        shl_debug_error("Size too large for platform: %lld > SIZE_MAX\n", (long long)size);
        return NULL;
    }
    
#ifdef SHL_MEM_DEBUG_VALID_WRITE
    // Check for overflow when adding  guard bytes
    const size_t guard_size = 8;
    if ((size_t)size > SIZE_MAX - guard_size) {
        shl_debug_error("Size overflow with guard bytes: %lld\n", (long long)size);
        return NULL;
    }
    
    ret = calloc(1, (size_t)size + guard_size);
    if (ret != NULL) {
        int8_t *check_ptr = ((int8_t *)ret) + size;
        /* magic number */
        check_ptr[0] = 0xff;
        check_ptr[1] = 0x23;
        check_ptr[2] = 0x33;
        check_ptr[3] = 0x44;
        check_ptr[4] = 0x45;
        check_ptr[5] = 0x55;
        check_ptr[6] = 0x67;
        check_ptr[7] = 0xff;
    }
#else
#ifdef SHL_USE_ATAT_MALLOC
    void *shl_atat_calloc(size_t n, size_t m);
    ret = shl_atat_calloc(1, (size_t)size);
#else
    ret = calloc(1, (size_t)size);
#endif
#endif

    if (ret == NULL) {
        shl_debug_error("cannot alloc memory\n");
    }
    
#ifdef SHL_MEM_DEBUG
    if (ret != NULL) {
        MUTEX_LOCK();
        
        if (shl_mem_map_insert(ret, size) != 0) {
            shl_debug_warning("Failed to track allocation\n");
        } else {
            shl_mem_alloc_debug_map.total_size += size;
            printf("shl_mem_alloc: total size = %ld, get size is %lld\n",
                   shl_mem_alloc_debug_map.total_size, size);
        }
        
        MUTEX_UNLOCK();
    }
#endif

    return ret;
}

/**
 * @brief Allocates and initializes memory for an array of elements
 * 
 * This function allocates memory for an array of nmemb elements, each of size bytes,
 * and initializes all bytes to zero. It includes overflow protection to prevent
 * integer overflow when calculating the total allocation size.
 * 
 * @param nmemb Number of elements to allocate
 * @param size Size in bytes of each element
 * 
 * @return Pointer to the allocated and zero-initialized memory on success,
 *         NULL if allocation fails or if integer overflow would occur
 * 
 * @note This function checks for potential integer overflow before performing
 *       the multiplication of nmemb and size parameters
 * @note The allocated memory is zero-initialized through shl_mem_alloc
 */
void *shl_mem_calloc(size_t nmemb, size_t size)
{
    // Check for overflow when multiplying
    if (nmemb != 0 && size > SIZE_MAX / nmemb) {
        shl_debug_error("calloc overflow: nmemb=%zu, size=%zu\n", nmemb, size);
        return NULL;
    }
    
    return shl_mem_alloc(nmemb * size);
}

/**
 * @brief Reallocates a memory block to a new size
 * 
 * This function allocates a new memory block of the specified size, copies data
 * from the original memory block, and frees the old memory block. If the original
 * pointer is NULL, it behaves like shl_mem_alloc(). The function copies the minimum
 * of the original size and new size to prevent buffer overflows.
 * 
 * @param ptr Pointer to the memory block to be reallocated. Can be NULL.
 * @param size New size in bytes for the memory block
 * @param orig_size Original size in bytes of the memory block pointed to by ptr.
 *                  If 0, the new size will be used for copying (with warning).
 * 
 * @return Pointer to the newly allocated memory block, or NULL if allocation fails.
 *         On allocation failure, the original memory block is not freed.
 * 
 * @warning If orig_size is 0, the function will copy 'size' bytes which may lead
 *          to undefined behavior if the original block is smaller than 'size'.
 * 
 * @note The original memory block is always freed on successful reallocation,
 *       even if the new size is smaller than the original size.
 */
void *shl_mem_realloc(void *ptr, size_t size, size_t orig_size)
{
    void *ret = shl_mem_alloc(size);
    if (!ptr) {
        return ret;
    }
    
    if (ret == NULL) {
        return NULL;  // Do not free old memory on failure
    }
    
    if (orig_size == 0) {
        shl_debug_warning(
            "New size(instead of original size) will be applied into memcpy, which may cause "
            "problems.\n");
        memcpy(ret, ptr, size);
    } else {
        // Copy the minimum of old and new size
        size_t copy_size = (orig_size < size) ? orig_size : size;
        memcpy(ret, ptr, copy_size);
    }
    
    shl_mem_free(ptr);
    return ret;
}

void *shl_mem_alloc_aligned(int64_t size, int aligned_bytes)
{
    void *ptr = NULL;
    
    // Check for negative size
    if (size <= 0 || aligned_bytes <= 0) {
        shl_debug_error("Invalid parameters: size=%lld, aligned_bytes=%d\n", 
                       (long long)size, aligned_bytes);
        return NULL;
    }
    
    // Check that aligned_bytes is a power of two
    if ((aligned_bytes & (aligned_bytes - 1)) != 0) {
        shl_debug_error("aligned_bytes must be power of 2, got %d\n", aligned_bytes);
        return NULL;
    }
    
#ifdef SHL_BUILD_RTOS
    // Check for overflow before addition
    // Need space for: size + alignment + pointer to original
    size_t extra_size = aligned_bytes + sizeof(void*);
    if ((size_t)size > SIZE_MAX - extra_size) {
        shl_debug_error("Integer overflow detected: size=%lld, aligned_bytes=%d\n",
                       (long long)size, aligned_bytes);
        return NULL;
    }
    
    size_t real_size = (size_t)size + extra_size;
    void *tptr = shl_mem_alloc(real_size);
    if (tptr == NULL) {
        return NULL;
    }

    // Compute the aligned address using the correct types
    uintptr_t raw_addr = (uintptr_t)tptr + sizeof(void*);
    uintptr_t mask = ~((uintptr_t)aligned_bytes - 1);
    uintptr_t aligned_addr = (raw_addr + aligned_bytes - 1) & mask;

    // Store the original pointer before the aligned address
    void **ptr_to_orig = (void**)(aligned_addr - sizeof(void*));
    *ptr_to_orig = tptr;
    
    ptr = (void*)aligned_addr;
#else
    if (aligned_bytes == 0) {
        aligned_bytes = getpagesize();
    }
    
    // posix_memalign requires alignment to be >= sizeof(void*)
    if (aligned_bytes < sizeof(void*)) {
        aligned_bytes = sizeof(void*);
    }
    
    int ret = posix_memalign(&ptr, aligned_bytes, size);
    if (ret || ptr == NULL) {
        shl_debug_error("cannot alloc aligned memory, error code: %d\n", ret);
    }
#endif
    
    return ptr;
}

/**
 * @brief Frees memory allocated by shl_mem_alloc and performs debug tracking if enabled
 * 
 * This function is a weak symbol that can be overridden by user implementations.
 * It provides memory deallocation with optional debug tracking and corruption detection.
 * 
 * @param ptr Pointer to the memory block to free. If NULL, the function returns immediately.
 * 
 * @details
 * When SHL_MEM_DEBUG is enabled:
 * - Tracks the deallocation in the debug map
 * - Updates total allocated memory size
 * - Marks the allocation entry as freed
 * - Warns if attempting to free an untracked pointer
 * 
 * When SHL_MEM_DEBUG_VALID_WRITE is also enabled:
 * - Checks guard bytes at the end of the allocated block for corruption
 * - Reports detailed information about any detected memory corruption
 * - Expected guard pattern: {0xff, 0x23, 0x33, 0x44, 0x45, 0x55, 0x67, 0xff}
 * 
 * The actual memory deallocation is performed using either:
 * - shl_atat_free() if SHL_USE_ATAT_MALLOC is defined
 * - Standard free() otherwise
 * 
 * @note This function is thread-safe when debug mode is enabled due to mutex protection
 * @warning Freeing the same pointer twice or freeing untracked pointers will generate warnings
 */
__attribute__((weak)) void shl_mem_free(void *ptr)
{
    if (ptr == NULL) {
        return;
    }
    
#ifdef SHL_MEM_DEBUG
    MUTEX_LOCK();
    
    int found = 0;
    for (int i = 0; i < shl_mem_alloc_debug_map.index; i++) {
        struct shl_mem_alloc_debug_element_ *e = shl_mem_alloc_debug_map.element + i;
        if (e->ptr == ptr && e->is_free == 0) {
            e->is_free = 1;
            shl_mem_alloc_debug_map.total_size -= e->size;
            printf("shl_mem_free: total size = %ld\n", shl_mem_alloc_debug_map.total_size);
            
#ifdef SHL_MEM_DEBUG_VALID_WRITE
            uint8_t *cptr = ptr + e->size;
            if ((cptr[0] == 0xff) && (cptr[1] == 0x23) && (cptr[2] == 0x33) && (cptr[3] == 0x44) &&
                (cptr[4] == 0x45) && (cptr[5] == 0x55) && (cptr[6] == 0x67) && (cptr[7] == 0xff)) {
                // Guard bytes intact
            } else {
                printf("shl_mem_free: invalid write %p\n", ptr);
                // Detailed information about corruption
                for (int j = 0; j < 8; j++) {
                    printf("  Guard byte %d: expected 0x%02x, got 0x%02x\n", 
                           j, ((uint8_t[]){0xff, 0x23, 0x33, 0x44, 0x45, 0x55, 0x67, 0xff})[j], 
                           cptr[j]);
                }
            }
#endif
            found = 1;
            break;
        }
    }
    
    if (!found) {
        shl_debug_warning("Attempting to free untracked pointer %p\n", ptr);
    }
    
    MUTEX_UNLOCK();
#endif

#ifdef SHL_USE_ATAT_MALLOC
    void shl_atat_free(void *f);
    shl_atat_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Frees memory that was allocated with alignment requirements in RTOS environment.
 * 
 * This function is used to free memory that was previously allocated with alignment
 * constraints. It retrieves the original unaligned pointer that was stored before
 * the aligned memory block and frees it using the standard memory deallocation function.
 * 
 * @param ptr Pointer to the aligned memory block to be freed. Can be NULL.
 * 
 * @note This function is only available when SHL_BUILD_RTOS is defined.
 * @note The ptr parameter should be a pointer that was returned by a corresponding
 *       aligned memory allocation function that stores the original pointer.
 * @note If ptr is NULL, the function returns immediately without performing any operation.
 * 
 * @warning This function assumes that the original pointer is stored in the memory
 *          location immediately before the aligned memory block (at offset -sizeof(void*)).
 *          Using this function with pointers not allocated by the corresponding aligned
 *          allocation function will result in undefined behavior.
 */
#ifdef SHL_BUILD_RTOS
void shl_mem_free_aligned(void *ptr)
{
    if (ptr == NULL) {
        return;
    }
    
    // Get the original pointer
    void **ptr_to_orig = (void**)((char*)ptr - sizeof(void*));
    void *orig_ptr = *ptr_to_orig;
    shl_mem_free(orig_ptr);
}
#endif
