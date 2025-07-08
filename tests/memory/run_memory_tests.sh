#!/bin/bash
#
# Script to deploy and run memory tests on LicheePi 4A
# Usage: ./run_memory_tests.sh [licheepi-ip]

set -e

# Configuration
LICHEEPI_IP=${1:-"192.168.1.44"}  # Default IP, can be overridden
LICHEEPI_USER="root"
REMOTE_DIR="/tmp/csi_nn2_tests"
TEST_BINARY="test_memory"
LOG_FILE="memory_test_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CSI-NN2 Memory Test Runner for LicheePi 4A${NC}"
echo "==========================================="

# Check if binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo -e "${YELLOW}Test binary not found. Building...${NC}"
    make clean
    make debug
fi

# Check if LicheePi is reachable
echo -e "\nChecking connection to ${LICHEEPI_IP}..."
if ping -c 1 -W 2 $LICHEEPI_IP > /dev/null 2>&1; then
    echo -e "${GREEN}✓ LicheePi is reachable${NC}"
else
    echo -e "${RED}✗ Cannot reach LicheePi at ${LICHEEPI_IP}${NC}"
    echo "Please check the IP address and network connection"
    exit 1
fi

# Create remote directory
echo -e "\nCreating remote directory..."
ssh ${LICHEEPI_USER}@${LICHEEPI_IP} "mkdir -p ${REMOTE_DIR}"

# Copy test binary
echo -e "Copying test binary..."
scp ${TEST_BINARY} ${LICHEEPI_USER}@${LICHEEPI_IP}:${REMOTE_DIR}/

# Create test runner script on remote
echo -e "Creating remote test script..."
cat << 'EOF' | ssh ${LICHEEPI_USER}@${LICHEEPI_IP} "cat > ${REMOTE_DIR}/run_tests.sh"
#!/bin/sh

echo "System Information:"
echo "==================="
uname -a
echo

echo "CPU Information:"
cat /proc/cpuinfo | grep -E "(processor|model name|cpu MHz)" | head -20
echo

echo "Memory Information:"
free -h
echo

echo "Running Memory Tests:"
echo "===================="

# Set CPU governor to performance for consistent results
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$cpu" ] && echo performance > "$cpu" 2>/dev/null || true
done

# Run the test
cd /tmp/csi_nn2_tests
./test_memory

# Reset CPU governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$cpu" ] && echo ondemand > "$cpu" 2>/dev/null || true
done
EOF

ssh ${LICHEEPI_USER}@${LICHEEPI_IP} "chmod +x ${REMOTE_DIR}/run_tests.sh"

# Run tests and capture output
echo -e "\n${YELLOW}Running tests on LicheePi 4A...${NC}\n"
ssh ${LICHEEPI_USER}@${LICHEEPI_IP} "${REMOTE_DIR}/run_tests.sh" | tee ${LOG_FILE}

# Check if tests passed
if grep -q "\[FAIL\]" ${LOG_FILE}; then
    echo -e "\n${RED}✗ Some tests failed!${NC}"
    echo "Failed tests:"
    grep "\[FAIL\]" ${LOG_FILE}
    exit_code=1
else
    echo -e "\n${GREEN}✓ All tests passed!${NC}"
    exit_code=0
fi

# Summary
echo -e "\nTest Summary:"
echo "============="
echo "Pass: $(grep -c "\[PASS\]" ${LOG_FILE} || echo 0)"
echo "Fail: $(grep -c "\[FAIL\]" ${LOG_FILE} || echo 0)"
echo "Warn: $(grep -c "\[WARN\]" ${LOG_FILE} || echo 0)"
echo
echo "Log saved to: ${LOG_FILE}"

# Optional: Copy any core dumps back
ssh ${LICHEEPI_USER}@${LICHEEPI_IP} "ls ${REMOTE_DIR}/core* 2>/dev/null" > /dev/null && {
    echo -e "\n${YELLOW}Core dumps found, copying back...${NC}"
    scp ${LICHEEPI_USER}@${LICHEEPI_IP}:${REMOTE_DIR}/core* . 2>/dev/null || true
}

exit $exit_code
