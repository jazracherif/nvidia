# CUDA Programming Guide 13.1 Examples

This folder contains CUDA kernel examples demonstrating various GPU programming concepts based on NVIDIA's CUDA Programming Guide version 13.1.

NOTE: The code, test, and readme are in part or in whole create with support from an AI coding agent.

## Table of Contents

- [Files Overview](#files-overview)
- [Testing](#testing)
- [Requirements](#requirements)
- [Common Issues](#common-issues)
- [Important CUDA Tips](#important-cuda-tips)
  - [Debugging with CUDA_LOG_FILE](#debugging-with-cuda_log_file)
- [Additional Resources](#additional-resources)

## Files Overview

For detailed descriptions of each CUDA example program, see [FILES.md](FILES.md).

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
