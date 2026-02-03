# Section 3: Performance Guidelines - CUDA Examples

This document contains examples from Section 3 of the [CUDA Programming Guide v13.1](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) - Performance Guidelines.

## Table of Contents

- [3.1.4 Programmatic Dependent Launch (`pdl.cu`)](#314-programmatic-dependent-launch-pdlcu)

---

## 3.1.4 Programmatic Dependent Launch (`pdl.cu`)

**Reference**: [CUDA Programming Guide Section 3.1.4 - Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#programmatic-dependent-kernel-launch)

**File**: [src/3.1.4-pdl.cu](src/3.1.4-pdl.cu)

Demonstrates Programmatic Dependent Launch (PDL), allowing a kernel to trigger another kernel's execution without returning to the host.

**Key Concepts:**
- Device-side kernel launch coordination
- `cudaTriggerProgrammaticLaunchCompletion()` - signals that dependent kernels can start
- `cudaGridDependencySynchronize()` - blocks until all dependent work completes
- `cudaLaunchAttributeProgrammaticStreamSerialization` - enables PDL on a kernel

**Architecture Pattern:**
```
Primary Kernel:
  ├─ Initial work
  ├─ cudaTriggerProgrammaticLaunchCompletion() ──┐
  └─ Overlapping work                             │
                                                  │
Secondary Kernel:                                 │
  ├─ Independent initialization                   │
  ├─ cudaGridDependencySynchronize() <────────────┘
  └─ Dependent work (waits for primary completion)
```

**Benefits:**
- Reduces kernel launch overhead by eliminating host round-trips
- Enables GPU-driven execution graphs
- Better overlap of independent operations
- Lower latency for dependent kernel launches

**Implementation Details:**

The primary kernel uses `cudaTriggerProgrammaticLaunchCompletion()` to signal when its initial work is done. This allows:
1. Secondary kernel to start its independent initialization immediately
2. Both kernels to execute overlapping independent work concurrently
3. Secondary kernel to synchronize and continue only when primary's dependent data is ready

The secondary kernel must be launched with the `cudaLaunchAttributeProgrammaticStreamSerialization` attribute via `cudaLaunchKernelEx()`.

**How to compile:**
```bash
nvcc -arch=native -o bin/pdl src/3.1.4-pdl.cu
```

**Requirements:**
- Compute capability 9.0+ (Hopper architecture or newer)
- CUDA Toolkit 12.0 or higher

**Use Cases:**
- Graph processing where next level depends on current level completion
- Iterative algorithms with device-side convergence checks
- Pipeline stages with complex dependencies
- Reducing PCIe latency in latency-sensitive applications

---

## Additional Resources

- [CUDA Programming Guide - Performance Guidelines](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#performance-guidelines)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
