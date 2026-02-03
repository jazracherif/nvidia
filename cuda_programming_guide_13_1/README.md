# CUDA Programming Guide 13 Deep Dive

This project is an attempt to better understand the CUDA programming model by diving deep into the [NVIDIA CUDA Programming Guide v13.1](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) documentation and testing various features through hands-on code implementation and debugging. Each example corresponds to specific sections of the Programming Guide, allowing for practical exploration of CUDA concepts from basic memory management to advanced performance optimization techniques.

The examples are tested on the [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) computer, profiled with tools like Nsight Systems, and documented with insights gained through experimentation and debugging sessions.

**NOTE:** The code, test, and readme are in part or in whole created with support from an AI coding agent.

## Table of Contents

- [Guide Overview](#guide-overview)
- [Additional Resources](#additional-resources)

## Guide Overview

### Documentation Structure

Examples are organized by Programming Guide sections:

- **[Section 2: Programming GPUs in CUDA](2-Programming-GPUs-in-CUDA.md)** - Core CUDA programming concepts
  - Memory management (unified vs. explicit)
  - Thread block clusters
  - Distributed shared memory
  - Asynchronous execution with events

- **[Section 3: Advanced CUDA](3-Advanced-CUDA.md)** - Advanced optimization techniques
  - Programmatic Dependent Launch (PDL)

### Quick File Reference

| File | Guide Section | Description |
|------|--------------|-------------|
| [2.1.3-memory-vecAdd.cu](src/2.1.3-memory-vecAdd.cu) | [2.1.3](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#heterogeneous-programming) | Unified vs. explicit memory management |
| [2.1.10-thread-cluster.cu](src/2.1.10-thread-cluster.cu) | [2.1.10](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#thread-block-clusters) | Thread block clusters (SM 9.0+) |
| [2.2.3.8-dist-memory.cu](src/2.2.3.8-dist-memory.cu) | [2.2.3.8](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#distributed-shared-memory) | Distributed shared memory histogram |
| [2.3.3.4-cuda-events.cu](src/2.3.3.4-cuda-events.cu) | [2.3.3.4](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#events) | Async streams with event coordination |
| [3.1.4-pdl.cu](src/3.1.4-pdl.cu) | [3.1.4](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#programmatic-dependent-launch-and-synchronization) | Programmatic Dependent Launch |

### Utilities

- **[query-device-property.cu](src/query-device-property.cu)** - GPU property query tool ([README](query-device-property-README.md))

### CUDA Development Tips

- **[Important CUDA Tips](cuda-tips-README.md)** - Debugging, profiling, and compilation techniques

### Requirements

- **CUDA Toolkit**: Version 11.0 or higher recommended
- **GPU**: NVIDIA GPU with compute capability 3.5 or higher
  - Note: Thread cluster example requires compute capability 9.0+
- **Compiler**: `nvcc` (NVIDIA CUDA Compiler)



## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
