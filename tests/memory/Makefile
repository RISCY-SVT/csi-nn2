# CSI-NN2 Memory Tests Makefile Documentation
# ==========================================
#
# This Makefile builds memory testing utilities for the CSI-NN2 neural network
# library specifically targeting the TH1520/C920 RISC-V processor on LicheePi 4A.
#
# TOOLCHAIN:
#   - Uses Xuantie RISC-V toolchain (riscv64-unknown-linux-gnu-) V3.1.0
#   - Targets rv64gcv0p7_zfh_xtheadc architecture with C920 tuning
#   - Generates static binaries for embedded deployment
#
# BUILD VARIANTS:
#   debug      - Development build with memory debugging and validation
#   release    - Optimized build for performance testing (-O3, NDEBUG)
#   minimal    - Size-optimized build without debug features (-Os)
#   multithread- Multi-threaded test without debug overhead
#
# DEBUGGING FEATURES:
#   - SHL_MEM_DEBUG: Enables memory allocation tracking
#   - SHL_MEM_DEBUG_VALID_WRITE: Validates memory write operations
#   - AddressSanitizer/ThreadSanitizer support (if toolchain supports)
#
# DEPENDENCIES:
#   - libshl_c920: CSI-NN2 library for C920 processor
#   - Static linking with pthread and math libraries
#   - Headers from CSI-NN2 installation directory
#
# DEPLOYMENT:
#   - Binaries are cross-compiled for RISC-V target
#   - Must be copied to LicheePi 4A device for execution
#   - No dynamic dependencies due to static linking
#
# ANALYSIS TOOLS:
#   disasm     - Generate disassembly for code analysis
#   info       - Display binary size, dependencies, and symbols
#   help       - Show available targets and usage
# Makefile for CSI-NN2 Memory Tests on TH1520/C920
# Target: LicheePi 4A

# Xuantie toolchain paths
TOOLCHAIN_PREFIX = riscv64-unknown-linux-gnu-
CC = $(TOOLCHAIN_PREFIX)gcc
LD = $(TOOLCHAIN_PREFIX)ld
AR = $(TOOLCHAIN_PREFIX)ar
OBJDUMP = $(TOOLCHAIN_PREFIX)objdump
SIZE = $(TOOLCHAIN_PREFIX)size

# Paths
CSI_NN2_INSTALL_DIR = ../../install_nn2/c920
INCLUDE_DIR = $(CSI_NN2_INSTALL_DIR)/include
LIB_DIR = $(CSI_NN2_INSTALL_DIR)/lib

# Compiler flags
CFLAGS = -static -O2 -g
CFLAGS += -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c920
CFLAGS += -I$(INCLUDE_DIR)
CFLAGS += -Wall -Wextra -Wno-unused-parameter

# Linker flags
LDFLAGS = -static -L$(LIB_DIR)
LDLIBS = -lshl_c920 -lpthread -lm

# Debug flags (будут добавлены только для debug цели)
DEBUG_FLAGS = -DSHL_MEM_DEBUG -DSHL_MEM_DEBUG_VALID_WRITE

# Targets
TARGET = test_memory
TARGET_MINIMAL = test_memory_minimal
TARGET_PERF = test_memory_perf
TARGET_MULTITHREAD = test_memory_multithread

.PHONY: all clean debug release info multithread

all: debug

# Debug build with all checks enabled
debug: CFLAGS += $(DEBUG_FLAGS) -g -O0
debug: $(TARGET)

# Release build for performance testing
release: CFLAGS += -O3 -DNDEBUG
release: $(TARGET_PERF)

# Minimal test without debug features
minimal: CFLAGS += -Os
# Multithread test without debug
multithread: CFLAGS += -O2
multithread: $(TARGET_MULTITHREAD)

$(TARGET_MULTITHREAD): test_memory_multithread.c
	@echo "Building multithread test without debug..."
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete. Binary size:"
	@$(SIZE) $@

$(TARGET): test_memory_th1520.c
	@echo "Building debug version..."
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete. Binary size:"
	@$(SIZE) $@

$(TARGET_PERF): test_memory_th1520.c
	@echo "Building performance version..."
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete. Binary size:"
	@$(SIZE) $@

$(TARGET_MINIMAL): test_memory_th1520.c
	@echo "Building minimal version..."
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete. Binary size:"
	@$(SIZE) $@

# Disassembly for analysis
disasm: $(TARGET)
	$(OBJDUMP) -d $(TARGET) > $(TARGET).S

# Generate info about binary
info: $(TARGET)
	@echo "=== Binary Information ==="
	@file $(TARGET)
	@echo
	@echo "=== Size Information ==="
	@$(SIZE) -A $(TARGET)
	@echo
	@echo "=== Library Dependencies ==="
	@$(TOOLCHAIN_PREFIX)readelf -d $(TARGET) | grep NEEDED || echo "No dynamic dependencies (static binary)"
	@echo
	@echo "=== Symbols ==="
	@$(TOOLCHAIN_PREFIX)nm $(TARGET) | grep -E "(shl_mem_|malloc|free)" | head -20

clean:
	rm -f $(TARGET) $(TARGET_MINIMAL) $(TARGET_PERF) *.o *.S

# Run tests on target
run: $(TARGET)
	@echo "Copy $(TARGET) to your LicheePi 4A and run:"
	@echo "  scp $(TARGET) root@<licheepi-ip>:/tmp/"
	@echo "  ssh root@<licheepi-ip> '/tmp/$(TARGET)'"

# Build test with AddressSanitizer (if supported by toolchain)
asan: CFLAGS += -fsanitize=address -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address
asan: $(TARGET)

# Build test with ThreadSanitizer (if supported by toolchain)
tsan: CFLAGS += -fsanitize=thread -fno-omit-frame-pointer
tsan: LDFLAGS += -fsanitize=thread
tsan: $(TARGET)

# Help
help:
	@echo "CSI-NN2 Memory Test Build System"
	@echo "================================"
	@echo "Targets:"
	@echo "  make debug    - Build with debug features (default)"
	@echo "  make release  - Build optimized version"
	@echo "  make minimal  - Build minimal version"
	@echo "  make disasm   - Generate disassembly"
	@echo "  make info     - Show binary information"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make run      - Show how to run on target"
	@echo "  make asan     - Build with AddressSanitizer"
	@echo "  make tsan     - Build with ThreadSanitizer"
	@echo "  make help     - Show this help message"
