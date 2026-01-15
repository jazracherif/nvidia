# CUDA Programming Guide 13.1 Examples

This folder contains CUDA kernel examples demonstrating various GPU programming concepts based on NVIDIA's CUDA Programming Guide version 13.1.

## Files Overview

### 1. `2.1.3-memory-vecAdd.cu`
**Vector Addition with Memory Management**

Demonstrates different memory management techniques for GPU programming:
- **Unified Memory**: Using `cudaMallocManaged()` for automatic memory management between host and device
- **Explicit Memory**: Manual memory allocation using `cudaMalloc()` and `cudaMemcpy()` for data transfers

**Features:**
- Implements a simple vector addition kernel (`vecAdd`)
- Compares GPU results with CPU serial computation for validation
- Shows best practices for memory allocation and cleanup
- Uses `CUDA_CHECK` macro for error handling

**How to compile and run:**
```bash
nvcc -o vecAdd 2.1.3-memory-vecAdd.cu
./vecAdd
```

**Expected output:**
```
Unified Memory: CPU and GPU answers match
Explicit Memory: CPU and GPU answers match
```

---

### 2. `2.1.10-thread-cluster.cu`
**Thread Cluster Programming**

Demonstrates the use of thread clusters, a feature for advanced GPU architectures (compute capability 9.0+).

**Features:**
- Uses `__cluster_dims__(2, 1, 1)` attribute to define compile-time cluster size
- Implements a simple increment kernel that adds 1 to each array element
- Uses unified memory for simplified memory management
- Shows cluster-based kernel launch syntax

**Important Note:**
This program requires a GPU with compute capability 9.0 or higher (e.g., Hopper architecture). It will fail to compile on older architectures with the error:
```
error: __cluster_dims__ is not supported for this GPU architecture
```

**How to compile and run:**
```bash
nvcc -o threadCluster 2.1.10-thread-cluster.cu
./threadCluster
```

---

### 3. `query-device-property.cu`
**GPU Device Property Query Tool**

A utility program to query and display GPU device properties, specifically focused on shared memory capabilities.

**Features:**
- Query shared memory per multiprocessor (`smpm`)
- Query shared memory per block (`smpb`)
- Query all supported properties (`all`)
- Supports multiple GPUs in the system
- Flexible command-line interface with colon-separated property list

**How to compile:**
```bash
nvcc -o query-device-property query-device-property.cu
```

**Usage:**
```bash
./query-device-property <prop1>:<prop2>:...
```

**Available properties:**
- `smpm` - sharedMemPerMultiprocessor
- `smpb` - sharedMemPerBlock
- `all` - all supported properties

**Examples:**
```bash
# Query shared memory per multiprocessor
./query-device-property smpm

# Query shared memory per block
./query-device-property smpb

# Query both properties
./query-device-property smpm:smpb

# Query all properties
./query-device-property all
```

**Example output:**
```
Found 1 CUDA device(s)

Device 0: NVIDIA GeForce RTX 3090
  Shared memory per multiprocessor: 102400 bytes (100.00 KB)
  Shared memory per block: 49152 bytes (48.00 KB)
```

---

## Testing

A test script is provided for the query tool:

```bash
chmod +x test-query-device-property.sh
./test-query-device-property.sh
```

This script will:
1. Compile the query-device-property program
2. Run multiple test cases to verify functionality
3. Display results for various property queries

---

## Requirements

- **CUDA Toolkit**: Version 11.0 or higher recommended
- **GPU**: NVIDIA GPU with compute capability 3.5 or higher
  - Note: Thread cluster example requires compute capability 9.0+
- **Compiler**: `nvcc` (NVIDIA CUDA Compiler)

## Common Issues

1. **Compilation errors about `__cluster_dims__`**: Your GPU architecture doesn't support thread clusters. This is expected for GPUs older than Hopper (compute capability < 9.0).

2. **No CUDA-capable devices found**: Ensure NVIDIA drivers are properly installed and your GPU is recognized by the system.

3. **Runtime errors**: Check that CUDA Toolkit version matches your driver version using `nvidia-smi`.

## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
