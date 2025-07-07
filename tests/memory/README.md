# CSI-NN2 Memory Tests

Comprehensive memory testing suite for CSI-NN2 library on RISC-V platforms, specifically designed for LicheePi 4A with TH1520 SoC (C920 cores).

## Overview

This test suite validates the memory management implementation in CSI-NN2, including:
- Basic allocation/deallocation
- Aligned memory allocation
- Buffer overflow detection (guard bytes)
- Thread safety
- Stress testing
- Memory debugging features

## Requirements

- **Target Hardware**: LicheePi 4A (TH1520 SoC with C920 cores)
- **Toolchain**: Xuantie RISC-V GCC (riscv64-unknown-linux-gnu-gcc) ≧ V3.1.0
- **Libraries**: CSI-NN2 library built for C920
- **OS**: Linux with pthread support

## Building the Tests

### Quick Start

```bash
# Build with debug features enabled (recommended for testing)
make debug

# Build optimized version for performance testing
make release

# Build minimal version without debug features
make minimal
```

### Build Options

| Target | Description | Use Case |
|--------|-------------|----------|
| `make debug` | Full debug build with memory tracking | Development and debugging |
| `make release` | Optimized build | Performance testing |
| `make minimal` | Minimal build | Quick functionality check |
| `make asan` | Build with AddressSanitizer | Memory error detection |
| `make tsan` | Build with ThreadSanitizer | Thread safety verification |

### Additional Commands

```bash
# Show binary information
make info

# Generate disassembly
make disasm

# Clean build artifacts
make clean

# Show help
make help
```

## Running the Tests

### Method 1: Automated Deployment (Recommended)

Use the provided script to automatically deploy and run tests on your LicheePi 4A:

```bash
# Make the script executable
chmod +x run_memory_tests.sh

# Run tests (replace with your device IP)
./run_memory_tests.sh 192.168.1.100
```

The script will:
- Check connectivity to the device
- Copy the test binary
- Set CPU governor to performance mode
- Run tests and capture output
- Save results to a timestamped log file

### Method 2: Manual Deployment

```bash
# Copy test binary to device
scp test_memory root@<device-ip>:/tmp/

# SSH into device
ssh root@<device-ip>

# Run tests
cd /tmp
./test_memory

# Or run with output redirection
./test_memory > memory_test_$(date +%Y%m%d_%H%M%S).log 2>&1
```

### Method 3: Quick Test

For a quick functionality check:

```bash
# Build and run quick test
gcc -static -O2 -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d \
    -I./include -L./lib quick_memory_test.c -lshl_c920 -o quick_test

scp quick_test root@<device-ip>:/tmp/
ssh root@<device-ip> /tmp/quick_test
```

## Test Coverage

### 1. Basic Memory Operations
- Standard allocation (`shl_mem_alloc`)
- Zero-size allocation handling
- Negative size allocation handling
- Large allocation support
- Memory initialization verification

### 2. Aligned Memory Allocation
- Power-of-2 alignments (8, 16, 32, 64, 128, 256, 512, 1024, 4096)
- Invalid alignment rejection
- Address alignment verification

### 3. Calloc and Realloc
- Zero initialization in `shl_mem_calloc`
- Overflow protection in size calculations
- Data preservation in `shl_mem_realloc`
- Size expansion and shrinking

### 4. Buffer Overflow Detection
- Guard byte verification (when `SHL_MEM_DEBUG_VALID_WRITE` is enabled)
- Corruption detection on free
- Detailed corruption reporting

### 5. Thread Safety
- Concurrent allocations from multiple threads
- CPU affinity for maximum contention
- Aligned allocation thread safety
- Race condition detection

### 6. Stress Testing
- 100,000 random allocation/free operations
- Memory leak detection
- Long-running stability test
- Random size allocations

### 7. Debug Features
- Memory allocation tracking
- Leak detection
- Memory map visualization

## Understanding Test Output

### Result Indicators

- **[PASS]** - Test completed successfully
- **[FAIL]** - Test failed, requires investigation
- **[WARN]** - Warning, may be expected behavior
- **[INFO]** - Informational message

### Example Output

```
CSI-NN2 Memory Test Suite for TH1520/C920
==========================================
System: Linux 5.10.113 riscv64

=== Test 1: Basic Memory Allocation ===
[PASS] Zero size allocation returns NULL
[PASS] Negative size allocation returns NULL
[PASS] Normal allocation successful
[PASS] Allocated memory is zeroed
[PASS] Memory write/read test passed
[PASS] Memory freed successfully
```

### Memory Status Information

The tests display memory usage statistics:
- VmRSS: Resident Set Size (physical memory)
- VmPeak: Peak virtual memory size
- VmSize: Current virtual memory size

## Debugging Failed Tests

### Enable Debug Mode

Build with debug flags to get detailed memory tracking:

```bash
make debug
./test_memory
```

### Check for Memory Corruption

If guard byte corruption is detected:
```
[WARN] Memory corruption detected at 0x3f9a8c0040!
  Guard byte 0: expected 0xff, got 0xde
  Guard byte 1: expected 0x23, got 0xad
```

This indicates a buffer overflow in the application code.

### Thread Safety Issues

For thread-related failures, use ThreadSanitizer:
```bash
make tsan
./test_memory
```

### Memory Leaks

Check the final memory map for unreleased allocations:
```
After freeing all:
  Total size = 0
```

Non-zero total size indicates a memory leak.

## Performance Considerations

### CPU Governor

The test script automatically sets CPU governor to performance mode:
```bash
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### NUMA Awareness

For multi-core systems, tests use CPU affinity to ensure proper stress testing across all cores.

## Customization

### Modifying Test Parameters

Edit `test_memory_th1520.c` to adjust:
- Number of threads
- Allocation sizes
- Iteration counts
- Delay between operations

### Adding New Tests

1. Add test function following the pattern:
```c
static void test_my_feature() {
    TEST_INFO("=== Test X: My Feature ===");
    // Test implementation
}
```

2. Call from `main()`:
```c
test_my_feature();
```

## Troubleshooting

### Build Errors

1. **Missing toolchain**: Ensure RISC-V toolchain is in PATH
2. **Library not found**: Check that CSI-NN2 is built and libs are in `./lib`
3. **Header not found**: Verify includes are in `./include`

### Runtime Errors

1. **Illegal instruction**: Wrong architecture flags, use `-march=rv64gcv0p7_zfh_xtheadc`
2. **Segmentation fault**: Enable debug build and check for memory corruption
3. **Assertion failed**: Check test output for specific failure details

### Connection Issues

1. **SSH timeout**: Verify device IP and network connectivity
2. **Permission denied**: Ensure root access or proper permissions
3. **No space left**: Check available space on target device

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Use provided TEST_* macros for consistent output
3. Include both positive and negative test cases
4. Document expected behavior
5. Test on actual hardware before submitting

## License

This test suite follows the same license as CSI-NN2 library (Apache-2.0).
