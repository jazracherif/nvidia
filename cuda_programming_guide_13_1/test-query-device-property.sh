#!/bin/bash

set -e

CUDA_FILE="query-device-property.cu"
EXECUTABLE="query-device-property"

echo "=========================================="
echo "Testing CUDA Device Property Query Program"
echo "=========================================="
echo ""

# Clean up any existing executable
if [ -f "$EXECUTABLE" ]; then
    echo "Removing existing executable..."
    rm "$EXECUTABLE"
fi

# Compile the CUDA program
echo "Compiling $CUDA_FILE..."
nvcc -o "$EXECUTABLE" "$CUDA_FILE"

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

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
