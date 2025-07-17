/*
 * Test to understand correct data access pattern in reordered buffer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csi_nn.h"
#include "shl_utils.h"
#include "rvv/rvv.h"

extern void shl_rvv_reorder_input_z16_fp16(__fp16 *b, __fp16 *sb, int k, int n, int ldx);

// Access element at (row, col) in reordered buffer
__fp16 get_reordered_element(const __fp16 *sb, int k, int n, int row, int col) {
    // Determine which block contains this column
    size_t offset = 0;
    int block_start = 0;
    int block_width = 0;
    
    // Find the block containing 'col'
    int temp_n = n;
    int temp_col = 0;
    
    while (temp_col <= col && temp_n > 0) {
        block_start = temp_col;
        
        if (temp_n >= 16) {
            block_width = 16;
        } else if (temp_n >= 8) {
            block_width = 8;
        } else if (temp_n >= 4) {
            block_width = 4;
        } else if (temp_n >= 2) {
            block_width = 2;
        } else {
            block_width = 1;
        }
        
        if (col < temp_col + block_width) {
            // Found the block
            break;
        }
        
        // Move to next block
        offset += k * block_width;
        temp_col += block_width;
        temp_n -= block_width;
    }
    
    // Position within block
    int col_in_block = col - block_start;
    
    // In reordered buffer, data is stored as:
    // [row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]
    // So for a given (row, col_in_block), the position is:
    // row * block_width + col_in_block
    
    return sb[offset + row * block_width + col_in_block];
}

int main() {
    printf("Test correct GEMM data access pattern\n");
    printf("=====================================\n");
    
    // Test with various n values
    int test_cases[][2] = {{3, 3}, {3, 5}, {3, 7}, {3, 9}, {5, 13}};
    
    for (int tc = 0; tc < 5; tc++) {
        int k = test_cases[tc][0];
        int n = test_cases[tc][1];
        
        printf("\n--- Test case: k=%d, n=%d ---\n", k, n);
        
        // Allocate and initialize
        __fp16 *B = (__fp16*)shl_mem_alloc(k * n * sizeof(__fp16));
        __fp16 *B_reordered = (__fp16*)shl_mem_alloc(k * n * sizeof(__fp16));
        
        // Fill with sequential values
        for (int i = 0; i < k * n; i++) {
            B[i] = (__fp16)(i * 0.1f);
        }
        
        // Print original
        printf("Original B:\n");
        for (int i = 0; i < k; i++) {
            printf("  row %d:", i);
            for (int j = 0; j < n; j++) {
                printf(" %5.2f", (float)B[i * n + j]);
            }
            printf("\n");
        }
        
        // Reorder
        shl_rvv_reorder_input_z16_fp16(B, B_reordered, k, n, n);
        
        // Print raw reordered buffer
        printf("\nReordered buffer (raw):");
        for (int i = 0; i < k * n; i++) {
            if (i % 8 == 0) printf("\n  ");
            printf("%5.2f ", (float)B_reordered[i]);
        }
        printf("\n");
        
        // Test access function
        printf("\nAccessing via get_reordered_element:\n");
        int errors = 0;
        for (int i = 0; i < k; i++) {
            printf("  row %d:", i);
            for (int j = 0; j < n; j++) {
                __fp16 val = get_reordered_element(B_reordered, k, n, i, j);
                printf(" %5.2f", (float)val);
                
                // Check if correct
                if (fabsf((float)val - (float)B[i * n + j]) > 0.001f) {
                    printf("!");
                    errors++;
                }
            }
            printf("\n");
        }
        
        printf("Errors: %d\n", errors);
        
        // Cleanup
        shl_mem_free(B);
        shl_mem_free(B_reordered);
    }
    
    return 0;
}
