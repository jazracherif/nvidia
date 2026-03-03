# AMD vs NVIDIA GPU Terminology Mapping

This table maps equivalent concepts between AMD (ROCm/HIP) and NVIDIA (CUDA) GPU programming and architecture.
Where a concept has no direct counterpart, **N/A** is noted with an explanation.

---

## Architecture Concepts

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| GPU compute architecture generation | **CDNA** (Compute DNA) | **SM architecture** (e.g., Hopper, Blackwell) | The generation and design of the GPU's compute architecture |
| GPU execution unit | **Compute Unit (CU)** | **Streaming Multiprocessor (SM)** | The fundamental independently-scheduled compute block on the GPU; contains ALUs, register file, schedulers, and local memory |
| Scalar ALU lane / arithmetic unit | **Stream Processor (SP)** | **CUDA Core** (FP32 core) | An individual FP32 arithmetic lane within a CU/SM; 64 per CU (AMD) vs 128 per SM (Blackwell) |
| Matrix multiply-accumulate unit | **Matrix Core (MFMA)** | **Tensor Core** | Dedicated hardware unit for matrix multiply-accumulate operations; used for AI/HPC mixed-precision |
| Thread group issued together | **Wavefront** (64 work-items) | **Warp** (32 threads) | The group of work-items/threads that execute in lockstep on a SIMD unit; AMD wavefronts are 2× wider |
| Single GPU thread | **Work-item** | **Thread** | One scalar execution context within a kernel |
| Group of threads / work-items sharing memory | **Workgroup** | **Thread Block (CTA)** | A cooperative group of work-items/threads that share LDS/shared memory and can synchronize |
| Grid of workgroups / thread blocks | **NDRange** | **Grid** | The total set of all workgroups/thread blocks dispatched for a kernel launch |
| Kernel dispatch | **Kernel dispatch (HIP)** | **Kernel launch (CUDA)** | Submitting a GPU kernel for execution |
| SIMD execution lane grouping | **SIMD-16 unit** (4 per CU) | **Warp scheduler / dispatch unit** | Sub-CU/sub-SM execution pipeline; AMD has 4 SIMD-16 units per CU (each 16 wide), NVIDIA has 4 warp schedulers per SM |
| GPU chiplet die | **GCD** (GPU Compute Die) | **GPC / Die** | Physical silicon die containing compute units (AMD uses chiplet packaging for MI300X: 3 GCDs) |
| Memory chiplet | **MCD** (Memory Cache Die) | **Memory die / MCM substrate** | Chiplet housing Infinity Cache and HBM interface logic (AMD); NVIDIA uses monolithic or TSV-integrated dies |
| GPU target identifier | **Target ID / GFX target** (e.g., gfx942) | **Compute Capability** (e.g., sm_90) | Identifies the hardware architecture and feature set for compilation |

---

## Memory & Storage

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| Per-CU/SM fast programmer-managed scratchpad | **LDS (Local Data Share)** | **Shared Memory (`__shared__`)** | Fast on-chip SRAM per CU/SM; programmer-explicitly managed; used for data reuse within a workgroup/thread block |
| Vector register file | **VGPR (Vector General Purpose Register)** | **Register file** | Per-lane (per-thread) register storage for arithmetic; 256 KB per CU (AMD) / 256 KB per SM (NVIDIA) |
| Scalar register file | **SGPR (Scalar General Purpose Register)** | **Uniform register / `__uniform__`** | Registers shared across all lanes in a wavefront/warp for control flow and address computation |
| Per-CU/SM data cache (transparent, HW-managed) | **L1 Cache** | **L1 Cache / Texture Cache** | Hardware-managed first-level data cache per CU/SM; not programmer-addressable |
| Last-level on-die GPU cache | **Infinity Cache** | **L2 Cache** | Large last-level cache on the GPU die shared across all CUs/SMs; reduces main memory traffic |
| Main GPU memory | **HBM (HBM3 / HBM3E)** | **HBM / GDDR** | High-bandwidth off-chip DRAM attached to the GPU; used for global tensors and model weights |
| Global memory address space | **Global Memory (flat address space)** | **Global Memory / Device Memory** | Largest but slowest GPU memory tier; addressable by all CUs/SMs |
| Memory accessible by all work-items in a workgroup | **LDS (Local Data Share)** | **Shared Memory** | See above; both share the same role as fast intra-workgroup/thread-block scratchpad |
| Constant memory | **Scalar Memory / SGPR broadcast** | **Constant Memory (`__constant__`)** | Read-only data broadcast efficiently to all lanes; AMD uses SGPR loads; NVIDIA caches in a dedicated constant cache |

---

## Interconnect & Multi-GPU

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| High-speed GPU-to-GPU interconnect | **Infinity Fabric / XGMI** | **NVLink** | Direct chip-to-chip interconnect bypassing PCIe; enables coherent GPU peer memory access within a node |
| GPU-to-GPU interconnect bandwidth (per node) | **XGMI bandwidth** (~896 GB/s, CDNA 4) | **NVLink bandwidth** (~900 GB/s, H100 SXM5) | All-to-all bidirectional GPU interconnect within a single server node |
| Multi-GPU switch fabric | **Infinity Switch** (planned) / XGMI direct mesh | **NVSwitch** | A switching ASIC that provides full all-to-all GPU bandwidth within a rack; NVIDIA ships NVSwitches; AMD plans Infinity Switch |
| PCIe-coupled add-in card form factor | **PCIe OAM adapter** (limited availability) | **SXM / PCIe card** | Physical packaging of the GPU accelerator; NVIDIA SXM = high-density server; OAM = AMD's server form factor |
| High-density server-form-factor GPU socket | **OAM (OCP Accelerator Module)** | **SXM (Server eXchange Module)** | Industry-standard (OAM) vs proprietary (SXM) socket enabling direct-attach high-power and high-bandwidth GPUs in compute nodes |
| GPU direct memory DMA for networking | **HIPDirect RDMA** | **GPUDirect RDMA** | Allows NICs to DMA directly to/from GPU memory without host CPU bounce buffers |
| Multi-node collective communications library | **RCCL (ROCm Collective Communications Library)** | **NCCL (NVIDIA Collective Communications Library)** | Ring/tree-based all-reduce, broadcast, scatter, gather for multi-GPU and multi-node training |

---

## Programming Model

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| GPU programming language / API | **HIP (Heterogeneous Interface for Portability)** | **CUDA** | The primary GPU programming model; HIP mirrors CUDA C++ syntax and can be ported with `hipify` tools |
| Compiler | **hipcc (wraps clang/llvm)** | **nvcc (wraps host compiler)** | The GPU kernel compiler toolchain |
| Runtime API | **HIP Runtime API** | **CUDA Runtime API** | C API for device management, memory allocation, kernel launch, streams, events |
| Driver API (low-level) | **HIP Driver API / ROCr (Runtime)** | **CUDA Driver API** | Low-level direct hardware management APIs |
| Kernel function qualifier | **`__global__`** (same) | **`__global__`** | Marks a function as a GPU kernel callable from the host |
| Device function qualifier | **`__device__`** (same) | **`__device__`** | Marks a function callable only from the GPU |
| Shared memory qualifier | **`__shared__`** (same) | **`__shared__`** | Declares a variable in LDS / shared memory |
| Synchronize all work-items in a workgroup | **`__syncthreads()`** (same via HIP) | **`__syncthreads()`** | Barrier synchronization for all threads in a thread block/workgroup |
| Thread index | **`threadIdx`, `blockIdx`, `blockDim`, `gridDim`** (same) | **`threadIdx`, `blockIdx`, `blockDim`, `gridDim`** | Built-in variables for identifying work-item position in the NDRange/grid |
| Asynchronous command queue | **HIP Stream** | **CUDA Stream** | Ordered sequence of GPU operations executing asynchronously relative to the host |
| Timing / synchronization event | **HIP Event** | **CUDA Event** | GPU-side timestamp markers used for profiling and synchronization |

---

## Software Ecosystem

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| GPU software stack / platform | **ROCm (Radeon Open Compute)** | **CUDA Toolkit** | Complete GPU compute software platform: compiler, runtime, libraries, profilers |
| Math / BLAS library | **rocBLAS / hipBLAS** | **cuBLAS** | Optimized Basic Linear Algebra Subprograms for GPU |
| FFT library | **rocFFT / hipFFT** | **cuFFT** | GPU-accelerated Fast Fourier Transform |
| Sparse matrix library | **rocSPARSE / hipSPARSE** | **cuSPARSE** | Sparse linear algebra routines |
| Random number generation | **rocRAND / hipRAND** | **cuRAND** | GPU random number generators |
| Deep learning primitives | **MIOpen** | **cuDNN** | Optimized DNN primitives: convolution, normalization, activation, pooling |
| Graph / model inference optimizer | **MIGraphX** | **TensorRT** | Optimizes and executes neural network graphs (operator fusion, quantization) |
| Profiler / performance analysis | **ROCProfiler, Omniperf** | **Nsight Systems, Nsight Compute** | GPU kernel profiling, roofline analysis, and performance counter tools |
| Debugger | **ROCgdb** | **cuda-gdb** | Source-level GPU kernel debugger |
| Multi-GPU partitioning / virtualization | **Accelerator Partitioning Mode** | **MIG (Multi-Instance GPU)** | Hardware-level partitioning of a single GPU into multiple isolated instances |

---

## Key Structural Differences

| Aspect | AMD | NVIDIA | Impact |
|---|---|---|---|
| Wavefront / warp width | **64 work-items** per wavefront | **32 threads** per warp | AMD wavefronts are 2× wider; divergent branches waste more lanes; but each MFMA covers more data |
| FP64 throughput | **FP64 = FP32 throughput** (CDNA 3, gfx942) | **FP64 = 1/2 FP32** (H100); **FP64 = 1/64 FP32** (consumer GPUs) | AMD CDNA GPUs have symmetric FP32/FP64; very strong for HPC/scientific computing |
| Last-level cache naming | **Infinity Cache** (on-die) | **L2 Cache** | AMD's Infinity Cache is larger than typical NVIDIA L2; reduces HBM traffic |
| Chiplet design | **GCD + MCD chiplets** (MI300X) | **Monolithic die or MCM** (H100 single die; GB200 dual-die) | AMD uses more aggressive chiplet packaging for large memory capacity |
| Coherency model | **Flat unified virtual address via Infinity Fabric** | **Separate GPU/CPU address spaces; ATS on Grace** | AMD EPYC + Instinct: CPU and GPU can access each other's memory coherently without explicit copies |
