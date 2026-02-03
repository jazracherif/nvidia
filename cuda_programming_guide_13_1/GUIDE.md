# CUDA Programming Guide Examples

This repository contains CUDA code examples from the [NVIDIA CUDA Programming Guide v13.1](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).

## Documentation Structure

Examples are organized by Programming Guide sections:

- **[Section 2: Programming GPUs in CUDA](2-Programming-GPUs-in-CUDA.md)** - Core CUDA programming concepts
  - Memory management (unified vs. explicit)
  - Thread block clusters
  - Distributed shared memory
  - Asynchronous execution with events

- **[Section 3: Advanced CUDA](3-Advanced-CUDA.md)** - Advanced optimization techniques
  - Programmatic Dependent Launch (PDL)

## Quick Reference

| File | Guide Section | Description |
|------|--------------|-------------|
| [2.1.3-memory-vecAdd.cu](src/2.1.3-memory-vecAdd.cu) | [2.1.3](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#heterogeneous-programming) | Unified vs. explicit memory management |
| [2.1.10-thread-cluster.cu](src/2.1.10-thread-cluster.cu) | [2.1.10](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#thread-block-clusters) | Thread block clusters (SM 9.0+) |
| [2.2.3.8-dist-memory.cu](src/2.2.3.8-dist-memory.cu) | [2.2.3.8](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#distributed-shared-memory) | Distributed shared memory histogram |
| [2.3.3.4-cuda-events.cu](src/2.3.3.4-cuda-events.cu) | [2.3.3.4](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#events) | Async streams with event coordination |
| [3.1.4-pdl.cu](src/3.1.4-pdl.cu) | [3.1.4](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#programmatic-dependent-launch-and-synchronization) | Programmatic Dependent Launch |

## Utilities

- **[query-device-property.cu](src/query-device-property.cu)** - GPU property query tool ([README](query-device-property-README.md))

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
