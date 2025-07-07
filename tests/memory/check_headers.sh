#!/bin/bash

# check_headers.sh - CSI-NN2 Header and Library Verification Script
#
# DESCRIPTION:
#   This script verifies the availability of required CSI-NN2 headers and libraries
#   for the C920 target architecture. It performs comprehensive checks including:
#   - Header file existence verification
#   - Library file availability check
#   - Minimal compilation test with RISC-V cross-compiler
#
# USAGE:
#   ./check_headers.sh
#
# REQUIREMENTS:
#   - CSI-NN2 installation at ../../install_nn2/c920
#   - riscv64-unknown-linux-gnu-gcc cross-compiler
#   - Required headers: shl_memory.h, shl_utils.h
#   - Required library: libshl_c920.a or libshl_c920.so
#
# CHECKS PERFORMED:
#   1. Verifies presence of shl_memory.h header file
#   2. Verifies presence of shl_utils.h header file
#   3. Checks for libshl_c920 library (static or shared)
#   4. Creates and compiles a minimal test program using:
#      - RISC-V 64-bit cross-compilation
#      - Vector extensions (rv64gcv0p7_zfh_xtheadc)
#      - CSI-NN2 memory allocation functions
#
# OUTPUT:
#   - Status messages for each check (OK/NOT FOUND)
#   - Library file listing if found
#   - Compilation success/failure status
#   - Cleanup of temporary test files
#
# EXIT STATUS:
#   Script continues execution regardless of individual check failures
#   to provide complete verification report

# Check if required headers are available
echo "Checking header files availability..."

CSI_NN2_INSTALL_DIR="../../install_nn2/c920"
INCLUDE_DIR="${CSI_NN2_INSTALL_DIR}/include"

# Check CSI-NN2 headers
echo -n "Checking shl_memory.h... "
if [ -f "${INCLUDE_DIR}/shl_memory.h" ]; then
    echo "OK"
else
    echo "NOT FOUND"
    echo "Expected at: ${INCLUDE_DIR}/shl_memory.h"
fi

echo -n "Checking shl_utils.h... "
if [ -f "${INCLUDE_DIR}/shl_utils.h" ]; then
    echo "OK"
else
    echo "NOT FOUND"
    echo "Expected at: ${INCLUDE_DIR}/shl_utils.h"
fi

# Check library
LIB_DIR="${CSI_NN2_INSTALL_DIR}/lib"
echo -n "Checking libshl_c920.a... "
if [ -f "${LIB_DIR}/libshl_c920.a" ] || [ -f "${LIB_DIR}/libshl_c920.so" ]; then
    echo "OK"
    ls -la ${LIB_DIR}/libshl_c920.*
else
    echo "NOT FOUND"
    echo "Expected at: ${LIB_DIR}/libshl_c920.a or .so"
fi

# Test compilation of a minimal program
echo -e "\nTesting minimal compilation..."
cat > test_minimal.c << 'EOF'
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <sys/utsname.h>
#include <limits.h>
#include "shl_memory.h"

int main() {
    struct utsname info;
    uname(&info);
    printf("System: %s\n", info.sysname);
    printf("SIZE_MAX: %zu\n", SIZE_MAX);
    
    void *ptr = shl_mem_alloc(100);
    if (ptr) {
        printf("Memory allocation works\n");
        shl_mem_free(ptr);
    }
    return 0;
}
EOF

echo "Compiling test_minimal.c..."
riscv64-unknown-linux-gnu-gcc -static -O2 \
    -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d \
    -I${INCLUDE_DIR} -L${LIB_DIR} \
    test_minimal.c -lshl_c920 -lm -o test_minimal 2>&1

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    rm -f test_minimal test_minimal.c
else
    echo "Compilation failed!"
    echo "Please check the error messages above."
fi
