# Important CUDA Tips

A collection of useful tips and techniques for CUDA development, debugging, and optimization discovered through hands-on experimentation.

## 1) Debugging with CUDA_LOG_FILE

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

## 2) Analyzing Kernel Resource Usage with -res-usage

The `-res-usage` flag in `nvcc` provides detailed information about resource consumption for each kernel during compilation. This helps optimize kernel performance by understanding register usage, memory requirements, and potential bottlenecks.

**Usage:**
```bash
nvcc -res-usage [other flags] -o <binary> <source.cu>
```

**Example:**
```bash
nvcc -res-usage -O2 -arch=native -o bin/2.3.3.4-cuda-events src/2.3.3.4-cuda-events.cu
```

**Output Analysis:**

```
ptxas info    : 48 bytes gmem
ptxas info    : Compiling entry function '_Z22computeIntensiveKernelPfi' for 'sm_121'
ptxas info    : Function properties for _Z22computeIntensiveKernelPfi
    32 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 32 bytes cumulative stack size
ptxas info    : Compile time = 7.603 ms
ptxas info    : Compiling entry function '_Z13vecInitRandomPfii' for 'sm_121'
ptxas info    : Function properties for _Z13vecInitRandomPfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers
ptxas info    : Compile time = 1.078 ms
ptxas info    : Compiling entry function '_Z6vecMulPfii' for 'sm_121'
ptxas info    : Function properties for _Z6vecMulPfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers
ptxas info    : Compile time = 0.622 ms
ptxas info    : Compiling entry function '_Z7vecInitPfii' for 'sm_121'
ptxas info    : Function properties for _Z7vecInitPfii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers
ptxas info    : Compile time = 0.476 ms
```

**Key Metrics Explained:**

1. **Global Memory (gmem)**: `48 bytes gmem`
   - Total size of kernel parameters across all kernels in the compilation unit
   - Each kernel's parameters (pointers, integers, etc.) are stored in constant memory
   - In this case: 4 kernels with parameters totaling 48 bytes
     - `vecInit(float*, int, int)`: 16 bytes (8 bytes (pointer) + 4 bytes (int) + 4 bytes (int))
     - `vecMul(float*, int, int)`: 16 bytes  
     - `vecInitRandom(float*, int, int)`: 16 bytes
     - `computeIntensiveKernel(float*, int)`: aligned to 16 bytes (8+4+padding)
   - This is stored in constant cache, not regular global memory

2. **Register Usage**: `Used 24 registers` (for `computeIntensiveKernel`)
   - Each thread uses 24 registers
   - Higher register usage can limit occupancy (threads per SM)
   - GPUs have limited registers per SM (e.g., 65536 on many architectures)
   - Lower register count allows more concurrent threads

3. **Stack Frame**: `32 bytes stack frame`
   - Local memory allocated on the stack per thread
   - Used for local arrays or when register spilling occurs
   - Stored in slower local memory (cached in L1/L2)

4. **Spill Stores/Loads**: `0 bytes spill stores, 0 bytes spill loads`
   - Register spilling occurs when a kernel uses more registers than available
   - Spilled registers are saved to local memory (slow)
   - Zero spills indicates efficient register allocation

5. **Barriers**: `used 0 barriers`
   - Number of `__syncthreads()` or similar synchronization points
   - Barriers can impact performance if overused

6. **Compile Time**: Time taken by `ptxas` to compile each kernel

**Performance Implications:**

- **`computeIntensiveKernel`**: Uses 24 registers with 32-byte stack - reasonable for complex math operations
- **`vecInitRandom`**: 10 registers, no stack - lightweight kernel
- **`vecMul` and `vecInit`**: 8 registers each - very efficient, allows maximum occupancy
- **No spilling**: Excellent - all variables fit in registers
- **No barriers**: Good for compute kernels that don't require thread synchronization

**Optimization Tips:**
- Monitor register usage to maximize occupancy
- Zero spills is ideal; non-zero spills suggest need for optimization
- Consider breaking up complex kernels if register count is too high
- Use `__launch_bounds__` to control register allocation

## 3) Understanding CUDA Compilation Pipeline

The `nvcc` compiler uses the `-time` flag to generate detailed timing information about each compilation phase, revealing the complex multi-stage process that transforms CUDA source code into an executable.

**Usage:**
```bash
nvcc -time <output_file> [other flags] -o <binary> <source.cu>
```

**Example:**
```bash
nvcc -time bin/compile.txt -O2 -arch=native -o bin/2.3.3.4-cuda-events src/2.3.3.4-cuda-events.cu
```

This generates a CSV file showing each compilation phase, its input files, output files, and execution time. By analyzing the dependencies between phase outputs and inputs, we can visualize the compilation pipeline.

**Raw Compilation Data (from `-time` flag):**

| Source File | Phase Name | Phase Input Files | Phase Output File | Arch | Tool | Metric | Unit |
|------------|------------|-------------------|-------------------|------|------|--------|------|
| src/2.3.3.4-cuda-events.cu | gcc (preprocessing 4) | src/2.3.3.4-cuda-events.cu | /tmp/tmpxft_00002814_00000000-6_2.3.3.4-cuda-events.cpp4.ii | | nvcc | 151.2190 | ms |
| src/2.3.3.4-cuda-events.cu | cudafe++ | /tmp/tmpxft_00002814_00000000-6_2.3.3.4-cuda-events.cpp4.ii | /tmp/tmpxft_00002814_00000000-7_2.3.3.4-cuda-events.cudafe1.cpp | compute_121 | nvcc | 414.9820 | ms |
| src/2.3.3.4-cuda-events.cu | gcc (preprocessing 1) | src/2.3.3.4-cuda-events.cu | /tmp/tmpxft_00002814_00000000-10_2.3.3.4-cuda-events.cpp1.ii | compute_121 | nvcc | 165.1220 | ms |
| src/2.3.3.4-cuda-events.cu | cicc | /tmp/tmpxft_00002814_00000000-10_2.3.3.4-cuda-events.cpp1.ii | /tmp/tmpxft_00002814_00000000-7_2.3.3.4-cuda-events.ptx | compute_121 | nvcc | 329.0240 | ms |
| src/2.3.3.4-cuda-events.cu | ptxas | /tmp/tmpxft_00002814_00000000-7_2.3.3.4-cuda-events.ptx | /tmp/tmpxft_00002814_00000000-11_2.3.3.4-cuda-events.cubin | sm_121 | nvcc | 32.1750 | ms |
| src/2.3.3.4-cuda-events.cu | fatbinary | /tmp/tmpxft_00002814_00000000-11_2.3.3.4-cuda-events.cubin | /tmp/tmpxft_00002814_00000000-4_2.3.3.4-cuda-events.fatbin | | nvcc | 1.8110 | ms |
| src/2.3.3.4-cuda-events.cu | gcc (compiling) | | /tmp/tmpxft_00002814_00000000-12_2.3.3.4-cuda-events.o | compute_121 | nvcc | 292.6390 | ms |
| src/2.3.3.4-cuda-events.cu | nvlink | /tmp/tmpxft_00002814_00000000-12_2.3.3.4-cuda-events.o | /tmp/tmpxft_00002814_00000000-13_2.3.3_dlink.cubin | sm_121 | nvcc | 2.7520 | ms |
| src/2.3.3.4-cuda-events.cu | fatbinary | /tmp/tmpxft_00002814_00000000-13_2.3.3_dlink.cubin | /tmp/tmpxft_00002814_00000000-9_2.3.3_dlink.fatbin | | nvcc | 1.2910 | ms |
| src/2.3.3.4-cuda-events.cu | gcc (compiling) | /usr/local/cuda/bin/crt/link.stub | /tmp/tmpxft_00002814_00000000-14_2.3.3_dlink.o | | nvcc | 12.1620 | ms |
| src/2.3.3.4-cuda-events.cu | gcc (linking) | /tmp/tmpxft_00002814_00000000-14_2.3.3_dlink.o /tmp/tmpxft_00002814_00000000-12_2.3.3.4-cuda-events.o | bin/2.3.3.4-cuda-events | | nvcc | 50.8550 | ms |
| | nvcc (driver) | | bin/2.3.3.4-cuda-events | | nvcc | 0.7740 | ms |

**CUDA Compilation Dependency Tree:**

```
CUDA Compilation Pipeline Dependency Tree
==========================================

Source: src/2.3.3.4-cuda-events.cu
│
├─[Branch 1: Host Code Processing]────────────────────────────────────────
│   │
│   ├─► `gcc` (preprocessing 4) [151.22 ms]
│   │   └─► Output: cpp4.ii
│   │
│   └─► `cudafe++` [414.98 ms]
│       └─► Input: cpp4.ii
│       └─► Output: cudafe1.cpp
│
├─[Branch 2: Device Code Processing]──────────────────────────────────────
│   │
│   ├─► `gcc` (preprocessing 1) [165.12 ms]
│   │   └─► Output: cpp1.ii
│   │
│   ├─► `cicc` (CUDA IR compiler) [329.02 ms]
│   │   └─► Input: cpp1.ii
│   │   └─► Output: .ptx
│   │
│   ├─► `ptxas` (PTX assembler) [32.18 ms]
│   │   └─► Input: .ptx
│   │   └─► Output: .cubin
│   │
│   └─► `fatbinary` [1.81 ms]
│       └─► Input: .cubin
│       └─► Output: .fatbin
│
├─[Merge: Object File Generation]─────────────────────────────────────────
│   │
│   └─► `gcc` (compiling) [292.64 ms]
│       └─► Inputs: cudafe1.cpp (implicit), .fatbin (implicit)
│       └─► Output: 2.3.3.4-cuda-events.o
│
├─[Device Linking]────────────────────────────────────────────────────────
│   │
│   ├─► `nvlink` (device linker) [2.75 ms]
│   │   └─► Input: 2.3.3.4-cuda-events.o
│   │   └─► Output: 2.3.3_dlink.cubin
│   │
│   ├─► `fatbinary` [1.29 ms]
│   │   └─► Input: 2.3.3_dlink.cubin
│   │   └─► Output: 2.3.3_dlink.fatbin
│   │
│   └─► `gcc` (compiling) [12.16 ms]
│       └─► Input: /usr/local/cuda/bin/crt/link.stub
│       └─► Output: 2.3.3_dlink.o
│
└─[Final Linking]─────────────────────────────────────────────────────────
    │
    └─► `gcc` (linking) [50.86 ms]
        └─► Inputs: 2.3.3_dlink.o + 2.3.3.4-cuda-events.o
        └─► Output: bin/2.3.3.4-cuda-events


Summary Flow Diagram:
=====================

                    src/2.3.3.4-cuda-events.cu
                              │
                    ┌─────────┴─────────┐
                    │                   │
           [Host Path]          [Device Path]
                    │                   │
              `gcc` (prep 4)      `gcc` (prep 1)
                [151 ms]            [165 ms]
                    │                   │
                cpp4.ii             cpp1.ii
                    │                   │
               `cudafe++`             `cicc`
                [415 ms]            [329 ms]
                    │                   │
              cudafe1.cpp             .ptx
                    │                   │
                    │                `ptxas`
                    │                [32 ms]
                    │                   │
                    │                .cubin
                    │                   │
                    │             `fatbinary`
                    │                [2 ms]
                    └──────►┬◄──────.fatbin
                            │
                      `gcc` (compiling)
                         [293 ms]
                            │
                   2.3.3.4-cuda-events.o
                            │
                        `nvlink`
                         [3 ms]
                            │
                     2.3.3_dlink.cubin
                            │
                      `fatbinary`
                         [1 ms]
                            │
                    2.3.3_dlink.fatbin
                            │
                  `gcc` (compiling link.stub)
                         [12 ms]
                            │
                     2.3.3_dlink.o
                            │
            ┌───────────────┴───────────────┐
            │                               │
     2.3.3_dlink.o             2.3.3.4-cuda-events.o
            │                               │
            └────────────►`gcc` (linking)◄──────┘
                           [51 ms]
                              │
                  bin/2.3.3.4-cuda-events
```

**Key Compilation Phases:**

1. **Host Path**: Preprocesses and transforms host-side CUDA code
   - `gcc (preprocessing 4)`: Initial preprocessing of source
   - `cudafe++`: CUDA Front End - transforms CUDA-specific syntax into standard C++

2. **Device Path**: Compiles GPU kernel code
   - `gcc (preprocessing 1)`: Preprocesses for device compilation
   - `cicc`: CUDA Internal Compiler - generates PTX (parallel thread execution) intermediate representation
   - `ptxas`: PTX Assembler - converts PTX to binary CUBIN (CUDA Binary) for specific GPU architecture
   - `fatbinary`: Packages CUBIN files into a fat binary supporting multiple architectures

3. **Merge & Link**: Combines host and device code
   - `gcc (compiling)`: Compiles host code and embeds device fat binary into object file
   - `nvlink`: Links device code across multiple object files
   - `gcc (linking)`: Final linking of host and device object files

**Critical Path**: The longest sequential path typically goes through `cudafe++` (415 ms) and `gcc (compiling)` (293 ms), making host-side C++ processing the bottleneck rather than device code generation.

**Total compilation time**: ~1.45 seconds

This visualization helps understand:
- Where compilation time is spent
- Which phases can run in parallel
- How changes to host vs device code affect build times
- The intermediate file dependencies in the build pipeline