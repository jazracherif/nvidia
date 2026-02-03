# GPU Device Property Query Tool

A utility program to query and display GPU device properties, specifically focused on shared memory capabilities and other device characteristics.

## Overview

This tool provides a flexible command-line interface for querying various CUDA device properties. It's useful for understanding GPU capabilities before running CUDA programs and for debugging device-specific issues.

## Features

- Query shared memory per multiprocessor (`smpm`)
- Query shared memory per block (`smpb`)
- Query all supported properties (`all`)
- Supports multiple GPUs in the system
- Flexible command-line interface with colon-separated property list
- Human-readable output with size conversions (bytes to KB)

## Compilation

```bash
nvcc -arch=native -o bin/query-device-property src/query-device-property.cu
```

## Usage

```bash
./bin/query-device-property <prop1>:<prop2>:...
```

## Available Properties

- `smpm` - sharedMemPerMultiprocessor
- `smpb` - sharedMemPerBlock
- `all` - all supported properties

## Examples

### Query shared memory per multiprocessor
```bash
./bin/query-device-property smpm
```

### Query shared memory per block
```bash
./bin/query-device-property smpb
```

### Query multiple properties
```bash
./bin/query-device-property smpm:smpb
```

### Query all properties
```bash
./bin/query-device-property all
```

## Example Output

```
Found 1 CUDA device(s)

Device 0: NVIDIA GeForce RTX 3090
  Shared memory per multiprocessor: 102400 bytes (100.00 KB)
  Shared memory per block: 49152 bytes (48.00 KB)
```

## Testing

A test script is provided to verify functionality:

```bash
chmod +x test-query-device-property.sh
./test-query-device-property.sh
```

This script will:
1. Compile the query-device-property program
2. Run multiple test cases to verify functionality
3. Display results for various property queries

## Requirements

- **CUDA Toolkit**: Version 11.0 or higher
- **GPU**: NVIDIA GPU with compute capability 3.5 or higher
- **Compiler**: `nvcc` (NVIDIA CUDA Compiler)

## Common Issues

1. **No CUDA-capable devices found**: Ensure NVIDIA drivers are properly installed and your GPU is recognized by the system using `nvidia-smi`.

2. **Compilation errors**: Verify that the CUDA Toolkit is properly installed and `nvcc` is in your PATH.

3. **Driver/toolkit mismatch**: Check that your CUDA Toolkit version is compatible with your NVIDIA driver version.
