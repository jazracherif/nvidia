#!/bin/bash

set -e

CUDA_FILE="src/query-device-property.cu"
EXECUTABLE="bin/query-device-property"

echo "=========================================="
echo "Testing CUDA Device Property Query Program"
echo "=========================================="
echo ""

# Create bin directory if it doesn't exist
mkdir -p bin

# Clean up any existing executable
if [ -f "$EXECUTABLE" ]; then
    echo "Removing existing executable..."
    rm "$EXECUTABLE"
fi

# Compile the CUDA program
echo "Compiling $CUDA_FILE..."
nvcc -O2 -arch=native -o "$EXECUTABLE" "$CUDA_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful"
    echo ""
else
    echo "✗ Compilation failed"
    exit 1
fi

# Test 1: Run without arguments (should show usage)
echo "=========================================="
echo "Test 1: Running without arguments"
echo "=========================================="
./"$EXECUTABLE" || true
echo ""

# Test 2: Query only sharedMemPerMultiprocessor
echo "=========================================="
echo "Test 2: Query sharedMemPerMultiprocessor (smpm)"
echo "=========================================="
./"$EXECUTABLE" smpm
echo ""

# Test 3: Query only sharedMemPerBlock
echo "=========================================="
echo "Test 3: Query sharedMemPerBlock (smpb)"
echo "=========================================="
./"$EXECUTABLE" smpb
echo ""

# Test 4: Query both properties
echo "=========================================="
echo "Test 4: Query both properties (smpm:smpb)"
echo "=========================================="
./"$EXECUTABLE" smpm:smpb
echo ""

# Test 5: Query with unknown property
echo "=========================================="
echo "Test 5: Query with unknown property (unknown)"
echo "=========================================="
./"$EXECUTABLE" unknown
echo ""

# Test 6: Query with mixed valid and invalid properties
echo "=========================================="
echo "Test 6: Mixed properties (smpm:unknown:smpb)"
echo "=========================================="
./"$EXECUTABLE" smpm:unknown:smpb
echo ""

# Test 7: Query all properties
echo "=========================================="
echo "Test 7: Query all properties (all)"
echo "=========================================="
./"$EXECUTABLE" all
echo ""

# Test 8: Query compute capability
echo "=========================================="
echo "Test 8: Query compute capability (cc)"
echo "=========================================="
./"$EXECUTABLE" cc
echo ""

# Test 9: Query toolkit and driver versions
echo "=========================================="
echo "Test 9: Query toolkit and driver versions"
echo "=========================================="
./"$EXECUTABLE" toolkit:driver
echo ""

# Test 10: Query mixed properties including new ones
echo "=========================================="
echo "Test 10: Mixed properties (cc:sm:smpm)"
echo "=========================================="
./"$EXECUTABLE" cc:sm:smpm
echo ""

# Test 11: Query register properties
echo "=========================================="
echo "Test 11: Query register properties (rgpm:rgpb)"
echo "=========================================="
./"$EXECUTABLE" rgpm:rgpb
echo ""

# Test 12: Query total constant memory
echo "=========================================="
echo "Test 12: Query total constant memory (tcm)"
echo "=========================================="
./"$EXECUTABLE" tcm
echo ""

# Test 13: Query L2 cache size
echo "=========================================="
echo "Test 13: Query L2 cache size (l2cs)"
echo "=========================================="
./"$EXECUTABLE" l2cs
echo ""

# Test 14: Query max threads per multiprocessor
echo "=========================================="
echo "Test 14: Query max threads per multiprocessor (mtpm)"
echo "=========================================="
./"$EXECUTABLE" mtpm
echo ""

# Test 15: Query max blocks per multiprocessor
echo "=========================================="
echo "Test 15: Query max blocks per multiprocessor (mbpm)"
echo "=========================================="
./"$EXECUTABLE" mbpm
echo ""

# Test 16: Query per-SM properties together
echo "=========================================="
echo "Test 16: Query per-SM properties (smpm:mtpm:mbpm)"
echo "=========================================="
./"$EXECUTABLE" smpm:mtpm:mbpm
echo ""

# Test 17: Query concurrent managed access
echo "=========================================="
echo "Test 17: Query concurrent managed access (cma)"
echo "=========================================="
./"$EXECUTABLE" cma
echo ""

# Test 18: Query pageable memory access
echo "=========================================="
echo "Test 18: Query pageable memory access (pma)"
echo "=========================================="
./"$EXECUTABLE" pma
echo ""

# Test 19: Query pageable memory access uses host page tables
echo "=========================================="
echo "Test 19: Query pageable memory access uses host page tables (pmahpt)"
echo "=========================================="
./"$EXECUTABLE" pmahpt
echo ""

# Test 20: Query memory access properties together
echo "=========================================="
echo "Test 20: Query memory access properties (cma:pma:pmahpt)"
echo "=========================================="
./"$EXECUTABLE" cma:pma:pmahpt
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
