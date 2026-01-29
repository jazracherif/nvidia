# CUDA Programming Guide 13.1 Examples

This folder contains CUDA kernel examples demonstrating various GPU programming concepts based on NVIDIA's CUDA Programming Guide version 13.1.

NOTE: The code, test, and readme are in part or in whole create with support from an AI coding agent.

## Table of Contents

- [Files Overview](#files-overview)
  - [1. `2.1.3-memory-vecAdd.cu`](#1-213-memory-vecaddcu)
  - [2. `2.1.10-thread-cluster.cu`](#2-2110-thread-clustercu)
  - [3. `query-device-property.cu`](#3-query-device-propertycu)
  - [4. `2.3.3.4-cuda-events.cu`](#4-2334-cuda-eventscu)
- [Testing](#testing)
- [Requirements](#requirements)
- [Common Issues](#common-issues)
- [Important CUDA Tips](#important-cuda-tips)
  - [Debugging with CUDA_LOG_FILE](#debugging-with-cuda_log_file)
- [Additional Resources](#additional-resources)

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
nvcc -arch=native -o bin/vecAdd src/2.1.3-memory-vecAdd.cu
./bin/vecAdd
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
nvcc -arch=native -o bin/threadCluster src/2.1.10-thread-cluster.cu
./bin/threadCluster
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
nvcc -arch=native -o bin/query-device-property src/query-device-property.cu
```

**Usage:**
```bash
./bin/query-device-property <prop1>:<prop2>:...
```

**Available properties:**
- `smpm` - sharedMemPerMultiprocessor
- `smpb` - sharedMemPerBlock
- `all` - all supported properties

**Examples:**
```bash
# Query shared memory per multiprocessor
./bin/query-device-property smpm

# Query shared memory per block
./bin/query-device-property smpb

# Query both properties
./bin/query-device-property smpm:smpb

# Query all properties
./bin/query-device-property all
```

**Example output:**
```
Found 1 CUDA device(s)

Device 0: NVIDIA GeForce RTX 3090
  Shared memory per multiprocessor: 102400 bytes (100.00 KB)
  Shared memory per block: 49152 bytes (48.00 KB)
```

---

### 4. `2.3.3.4-cuda-events.cu`
**CUDA Events and Asynchronous Stream Execution**

Demonstrates advanced stream management using CUDA events to coordinate asynchronous operations across multiple streams.

**Concept:**
This program illustrates how CUDA events enable fine-grained synchronization between operations on different streams without blocking the host CPU. The key technique shown is:
1. Launching kernels on one stream (stream1)
2. Recording an event after a specific kernel completes
3. Using `cudaEventQuery()` to non-blockingly check if that kernel has finished
4. Starting an asynchronous memory copy on a separate stream (stream2) as soon as the event signals completion
5. Continuing CPU work concurrently with GPU operations

**Architecture:**
```
Stream 1:  [vecInit] --EVENT--> [vecInitRandom] --> [computeIntensiveKernel]
                        ↓
                    (query event)
                        ↓
Stream 2:               [cudaMemcpyAsync D→H]

CPU:       [doNextChunkOfCPUWork...] (runs concurrently)
```

**Features:**
- **Dual Stream Execution**: Separates compute (stream1) from data transfer (stream2)
- **Event-Based Coordination**: Uses `cudaEventRecord()` and `cudaEventQuery()` to trigger actions without blocking
- **CPU/GPU Overlap**: CPU continues working while GPU kernels execute
- **Compute Kernels**:
  - `vecInit`: Initializes array with a constant value
  - `vecInitRandom`: Populates array with pseudo-random values
  - `computeIntensiveKernel`: Performs heavy mathematical operations (sin, cos, sqrt, exp, log, pow)
- **Pinned Memory**: Uses `cudaMallocHost()` for faster async transfers
- **Comprehensive Error Checking**: All CUDA calls wrapped in `CUDA_CHECK` macro

**How it works:**
1. Allocate device and pinned host memory
2. Create two streams for parallel execution
3. Launch `vecInit` kernel on stream1 and immediately record an event
4. Launch additional compute kernels on stream1
5. While GPU is busy, CPU polls the event in a non-blocking loop
6. As soon as the event indicates `vecInit` completed, start async D2H copy on stream2
7. CPU continues its work while both streams execute concurrently
8. Synchronize both streams before cleanup

**Why this matters:**
- **Overlap**: Demonstrates how to overlap computation with data transfer
- **CPU Utilization**: CPU isn't blocked waiting for GPU
- **Efficiency**: Multiple operations execute simultaneously across streams
- **Real-world Pattern**: Common in production pipelines where you want to copy results back as soon as they're ready, without waiting for all GPU work to finish

**How to compile and run:**
```bash
nvcc -O2 -arch=native -o bin/2.3.3.4-cuda-events src/2.3.3.4-cuda-events.cu
./bin/2.3.3.4-cuda-events
```

**Expected output:**
```
work_left 5
work_left 4
work_left 3
work_left 2
work_left 1
start async copy
work_left 0
```

**Profile with Nsight Systems:**
```bash
nsys profile -o 2.3.3.4-cuda-events ./bin/2.3.3.4-cuda-events
```
The timeline will show stream1 and stream2 executing concurrently, with the memory copy starting before all compute kernels finish.

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

## Important CUDA Tips

### Debugging with CUDA_LOG_FILE

The `CUDA_LOG_FILE` environment variable redirects CUDA driver errors and warnings to a specified file, making it easier to debug kernel launch errors and other runtime issues without cluttering your terminal output.

**Description:**
When CUDA encounters runtime errors (such as invalid kernel launch parameters, memory access violations, or API errors), the driver can log detailed diagnostic information. By setting `CUDA_LOG_FILE`, you capture these messages in a file for later analysis.

**Usage:**
```bash
# Set the log file and run your program
CUDA_LOG_FILE=cudaLog.txt ./bin/program

# Or export it for multiple runs
export CUDA_LOG_FILE=cudaLog.txt
./bin/program
```

**Combine with other debugging options:**
```bash
# Enable synchronous kernel launches and log to file
CUDA_LOG_FILE=cudaLog.txt CUDA_LAUNCH_BLOCKING=1 ./bin/program
```

**Example output in cudaLog.txt:**
```
[14:38:03.928][280411609614528][CUDA][E] One or more of block dimensions of (2256,1,1) exceeds correspsonding maximum value of (1024,1024,64)
[14:38:03.928][280411609614528][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

This is particularly useful for:
- Identifying kernel launch configuration errors
- Debugging asynchronous operations where errors may not appear immediately
- Capturing error logs from long-running programs
- Analyzing intermittent issues that are hard to reproduce

## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
