# CSI-NN2 Multithread Safety Test Makefile
# Target: TH1520/C920 RISC-V processor on LicheePi 4A

# Toolchain
TOOLCHAIN_PREFIX = riscv64-unknown-linux-gnu-
CC = $(TOOLCHAIN_PREFIX)gcc
SIZE = $(TOOLCHAIN_PREFIX)size
NM = $(TOOLCHAIN_PREFIX)nm
OBJDUMP = $(TOOLCHAIN_PREFIX)objdump

# Paths
CSI_NN2_INSTALL_DIR = ../../install_nn2/c920
INCLUDE_DIR = $(CSI_NN2_INSTALL_DIR)/include
LIB_DIR = $(CSI_NN2_INSTALL_DIR)/lib

# Common flags
CFLAGS = -static -I$(INCLUDE_DIR)
CFLAGS += -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c920
CFLAGS += -Wall -Wextra -Wno-unused-parameter
CFLAGS += -fopenmp -pthread -D_REENTRANT

LDFLAGS = -static -L$(LIB_DIR)
LDLIBS = -lshl_c920 -lpthread -lm -lgomp

# Targets
TARGET = test_multithread_safety
TARGET_DEBUG = test_multithread_safety_debug

# Default target
.PHONY: all
all: debug

# Debug build (default)
.PHONY: debug
debug: CFLAGS += -O0 -g3 -ggdb -DDEBUG_THREADS
debug: $(TARGET_DEBUG)

# Release build
.PHONY: release
release: CFLAGS += -O2 -DNDEBUG
release: $(TARGET)

# Build rules
$(TARGET): test_multithread_safety.c
	@echo "Building release version..."
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@$(SIZE) $@

$(TARGET_DEBUG): test_multithread_safety.c
	@echo "Building debug version..."
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)
	@$(SIZE) $@

# Analysis targets
.PHONY: info
info: $(TARGET_DEBUG)
	@echo "=== Binary Information ==="
	@file $(TARGET_DEBUG)
	@echo
	@echo "=== Size ==="
	@$(SIZE) $(TARGET_DEBUG)
	@echo
	@echo "=== Thread-related symbols ==="
	@$(NM) $(TARGET_DEBUG) | grep -E "(shl_multithread_|pthread_|atomic_)" | head -20

.PHONY: disasm
disasm: $(TARGET_DEBUG)
	$(OBJDUMP) -d $(TARGET_DEBUG) > $(TARGET_DEBUG).S
	@echo "Disassembly saved to $(TARGET_DEBUG).S"

# Clean
.PHONY: clean
clean:
	rm -f $(TARGET) $(TARGET_DEBUG) *.o *.S

# Deployment
.PHONY: deploy
deploy: $(TARGET_DEBUG)
	@echo "Copy to target:"
	@echo "  scp $(TARGET_DEBUG) root@<licheepi-ip>:/tmp/"
	@echo
	@echo "Run on target:"
	@echo "  ssh root@<licheepi-ip> '/tmp/$(TARGET_DEBUG)'"
	@echo
	@echo "Run with custom thread count:"
	@echo "  ssh root@<licheepi-ip> 'OMP_NUM_THREADS=4 /tmp/$(TARGET_DEBUG)'"

# Help
.PHONY: help
help:
	@echo "Multithread Safety Test Build"
	@echo "============================"
	@echo "Targets:"
	@echo "  make debug    - Build with debugging (default)"
	@echo "  make release  - Build optimized version"
	@echo "  make info     - Show binary information"
	@echo "  make disasm   - Generate disassembly"
	@echo "  make deploy   - Show deployment instructions"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help"