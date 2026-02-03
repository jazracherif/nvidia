# Files Overview

This repository contains CUDA code examples from the [NVIDIA CUDA Programming Guide v13.1](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).

## Documentation Structure

Examples are organized by Programming Guide sections:

- **[Section 2: Programming Interface](FILES-SECTION-2.md)** - Core CUDA programming concepts
  - Memory management (unified vs. explicit)
  - Thread block clusters
  - Distributed shared memory
  - Asynchronous execution with events

- **[Section 3: Performance Guidelines](FILES-SECTION-3.md)** - Advanced optimization techniques
  - Programmatic Dependent Launch (PDL)

- **[Utilities](#utilities)** - Helper programs for device queries

## Quick Reference

| File | Guide Section | Description |
|------|--------------|-------------|
| [2.1.3-memory-vecAdd.cu](src/2.1.3-memory-vecAdd.cu) | [2.1.3](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#heterogeneous-programming) | Unified vs. explicit memory management |
| [2.1.10-thread-cluster.cu](src/2.1.10-thread-cluster.cu) | [2.1.10](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#thread-block-clusters) | Thread block clusters (SM 9.0+) |
| [2.2.3.8-dist-memory.cu](src/2.2.3.8-dist-memory.cu) | [2.2.3.8](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#distributed-shared-memory) | Distributed shared memory histogram |
| [2.3.3.4-cuda-events.cu](src/2.3.3.4-cuda-events.cu) | [2.3.3.4](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#events) | Async streams with event coordination |
| [3.1.4-pdl.cu](src/3.1.4-pdl.cu) | [3.1.4](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#programmatic-dependent-launch-and-synchronization) | Programmatic Dependent Launch |
| [query-device-property.cu](src/query-device-property.cu) | Utility | GPU property query tool |

## Utilities

### `query-device-property.cu`
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
