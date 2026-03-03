# AMD vs NVIDIA GPU Terminology Mapping

This table maps equivalent concepts between AMD (ROCm/HIP) and NVIDIA (CUDA) GPU programming and architecture.

Where a concept has no direct counterpart, **N/A** is noted with an explanation.

---

## Architecture Concepts

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| GPU compute architecture generation | **CDNA** (Compute DNA) | **SM architecture** (e.g., Hopper, Blackwell) | The generation and design of the GPU's compute architecture. |
| GPU execution unit | **Compute Unit (CU)** | **Streaming Multiprocessor (SM)** | The fundamental independently-scheduled compute block on the GPU; contains ALUs, register file, schedulers, and local memory. |
| Scalar ALU lane / arithmetic unit | **Stream Processor (SP)** | **CUDA Core** (FP32 core) | An individual FP32 arithmetic lane within a CU/SM; 64 per CU (AMD) vs 128 per SM (Blackwell). |
| Matrix multiply-accumulate unit | **Matrix Core (MFMA)** | **Tensor Core** | Dedicated hardware unit for matrix multiply-accumulate operations; used for AI/HPC mixed-precision. |
| Thread group issued together | **Wavefront** (64 work-items) | **Warp** (32 threads) | The group of work-items/threads that execute in lockstep on a SIMD unit; AMD wavefronts are 2× wider. |
| Single GPU thread | **Work-item** | **Thread** | One scalar execution context within a kernel. |
| Group of threads / work-items sharing memory | **Workgroup** | **Thread Block (CTA)** | A cooperative group of work-items/threads that share LDS/shared memory and can synchronize. |
| Grid of workgroups / thread blocks | **NDRange** | **Grid** | The total set of all workgroups/thread blocks dispatched for a kernel launch. |
| Kernel dispatch | **Kernel dispatch (HIP)** | **Kernel launch (CUDA)** | Submitting a GPU kernel for execution. |
| SIMD execution lane grouping | **SIMD-16 unit** (4 per CU) | **SM Partition / Processing Block** | Sub-CU/sub-SM execution pipeline containing its own instruction buffer, scheduler, and cores. |
| GPU compute chiplet / die | **XCD** (Accelerator Complex Die) / **GCD** | **Compute Die** | Physical silicon die containing compute units. CDNA 3 uses XCDs; older architectures used GCDs. NVIDIA uses full Compute Dies (e.g., in dual-die GB200). |
| Infrastructure / memory routing die | **IOD (I/O Die) + Base Die** | **Memory die / MCM substrate** | Silicon housing the interconnect, memory controllers, and structural foundation for the compute chiplets. |
| GPU target identifier | **Target ID / GFX target** (e.g., `gfx942`) | **Compute Capability** (e.g., `sm_90`) | Identifies the hardware architecture and feature set for compilation. |

---

## Memory & Storage

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| Per-CU/SM fast programmer-managed scratchpad | **LDS (Local Data Share)** | **Shared Memory (`__shared__`)** | Fast on-chip SRAM per CU/SM, explicitly managed by the programmer. Accessible by all threads in a workgroup/thread block for data reuse. |
| Vector register file | **VGPR (Vector General Purpose Register)** | **Register file** | Per-lane (per-thread) register storage for arithmetic. |
| Scalar register file | **SGPR (Scalar General Purpose Register)** | **Uniform register / `__uniform__`** | Registers shared across all lanes in a wavefront/warp for control flow and address computation. |
| Per-CU/SM data cache (transparent) | **L1 Cache** | **L1 Cache / Texture Cache** | Hardware-managed first-level data cache per CU/SM; not programmer-addressable. |
| Last-level on-die GPU cache | **Infinity Cache** | **L2 Cache** | Large last-level cache on the GPU die shared across all CUs/SMs; reduces main memory traffic. |
| Main GPU memory | **HBM (HBM3 / HBM3E)** | **HBM / GDDR** | High-bandwidth off-chip DRAM attached to the GPU; used for global tensors and model weights. |
| Global memory address space | **Global Memory** (flat address space) | **Global Memory / Device Memory** | Largest but slowest GPU memory tier; addressable by all CUs/SMs. |
| Constant memory hardware / cache | **Scalar Data Cache (K$) / SGPR broadcast** | **Constant Cache / Constant Memory (`__constant__`)** | Read-only data broadcast efficiently to all lanes. AMD leverages a dedicated Scalar Data Cache and SGPRs; NVIDIA uses a dedicated Constant Cache. |



---

## Interconnect & Multi-GPU

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| High-speed GPU-to-GPU interconnect | **Infinity Fabric / XGMI** | **NVLink** | Direct chip-to-chip interconnect bypassing PCIe; enables coherent GPU peer memory access within a node. |
| GPU-to-GPU interconnect bandwidth | **XGMI bandwidth** | **NVLink bandwidth** | All-to-all bidirectional GPU interconnect within a single server node. |
| Multi-GPU switch fabric | **Infinity Switch** (planned) / XGMI direct mesh | **NVSwitch** | A switching ASIC that provides full all-to-all GPU bandwidth within a rack. |
| High-density server accelerator form factor | **OAM (OCP Accelerator Module)** | **SXM (Server eXchange Module)** | Industry-standard (OAM) vs proprietary (SXM) socket enabling direct-attach high-power and high-bandwidth GPUs in compute nodes. |
| GPU direct memory DMA for networking | **ROCm RDMA** | **GPUDirect RDMA** | Allows NICs to DMA directly to/from GPU memory without host CPU bounce buffers. |
| Multi-node collective communications | **RCCL (ROCm Collective Communications Library)** | **NCCL (NVIDIA Collective Communications Library)** | Ring/tree-based all-reduce, broadcast, scatter, gather for multi-GPU and multi-node training. |

---

## Programming Model

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| GPU programming language / API | **HIP (Heterogeneous Interface for Portability)** | **CUDA** | The primary GPU programming model; HIP mirrors CUDA C++ syntax and can be ported with `hipify` tools. |
| Compiler | **`amdclang++`** (legacy: `hipcc`) | **nvcc** | The GPU kernel compiler toolchain. Modern ROCm heavily favors directly invoking the LLVM-based `amdclang++`. |
| Runtime API | **HIP Runtime API** | **CUDA Runtime API** | C API for device management, memory allocation, kernel launch, streams, events. |
| Driver API (low-level) | **HIP Driver API / ROCr (Runtime)** | **CUDA Driver API** | Low-level direct hardware management APIs. |
| Kernel function qualifier | **`__global__`** | **`__global__`** | Marks a function as a GPU kernel callable from the host. |
| Device function qualifier | **`__device__`** | **`__device__`** | Marks a function callable only from the GPU. |
| Shared memory qualifier | **`__shared__`** | **`__shared__`** | Declares a variable in LDS / shared memory. |
| Synchronize all work-items in a block | **`__syncthreads()`** | **`__syncthreads()`** | Barrier synchronization for all threads in a thread block/workgroup. |
| Thread index variables | **`threadIdx`, `blockIdx`, `blockDim`, `gridDim`** | **`threadIdx`, `blockIdx`, `blockDim`, `gridDim`** | Built-in variables for identifying work-item position in the NDRange/grid. |
| Asynchronous command queue | **HIP Stream** | **CUDA Stream** | Ordered sequence of GPU operations executing asynchronously relative to the host. |
| Timing / synchronization event | **HIP Event** | **CUDA Event** | GPU-side timestamp markers used for profiling and synchronization. |

---

## Software Ecosystem

| Generic Concept | AMD Term | NVIDIA Term | Description |
|---|---|---|---|
| GPU software stack / platform | **ROCm (Radeon Open Compute)** | **CUDA Toolkit** | Complete GPU compute software platform: compiler, runtime, libraries, profilers. |
| Math / BLAS library | **rocBLAS / hipBLAS** | **cuBLAS** | Optimized Basic Linear Algebra Subprograms for GPU. |
| FFT library | **rocFFT / hipFFT** | **cuFFT** | GPU-accelerated Fast Fourier Transform. |
| Sparse matrix library | **rocSPARSE / hipSPARSE** | **cuSPARSE** | Sparse linear algebra routines. |
| Random number generation | **rocRAND / hipRAND** | **cuRAND** | GPU random number generators. |
| Deep learning primitives | **MIOpen** | **cuDNN** | Optimized DNN primitives: convolution, normalization, activation, pooling. |
| Graph / model inference optimizer | **MIGraphX** | **TensorRT** | Optimizes and executes neural network graphs (operator fusion, quantization). |
| Profiler / performance analysis | **ROCProfiler, Omniperf** | **Nsight Systems, Nsight Compute** | GPU kernel profiling, roofline analysis, and performance counter tools. |
| Debugger | **ROCgdb** | **cuda-gdb** | Source-level GPU kernel debugger. |
| Multi-GPU partitioning / virtualization | **Spatial Partitioning (SPX instances)** | **MIG (Multi-Instance GPU)** | Hardware-level partitioning of a single GPU into multiple isolated instances (e.g., for multi-tenant inference). |

---

## Key Structural Differences

| Aspect | AMD | NVIDIA | Impact |
|---|---|---|---|
| Wavefront / warp width | **64 work-items** per wavefront | **32 threads** per warp | AMD wavefronts are 2× wider; divergent branches waste more lanes; but each MFMA instruction covers more data. |
| FP64 throughput | **FP64 = FP32 throughput** (CDNA 3, `gfx942`) | **FP64 = 1/2 FP32** (H100) | AMD CDNA GPUs have symmetric FP32/FP64; heavily optimized for traditional HPC/scientific computing. |
| Last-level cache naming | **Infinity Cache** (on-die) | **L2 Cache** | AMD's Infinity Cache design is typically larger than standard NVIDIA L2, heavily mitigating HBM traffic. |
| Chiplet design | **XCD/IOD chiplets** (MI300X) | **Monolithic die or MCM** (H100/GB200) | AMD utilizes aggressive 3D packaging and chiplets for yield and memory capacity, whereas NVIDIA leans toward larger monolithic or dual-die packages. |
| Coherency model | **Flat unified virtual address via Infinity Fabric** | **Separate GPU/CPU address spaces; ATS on Grace** | AMD APU designs (like MI300A) allow CPU and GPU to access each other's memory completely coherently without explicit PCIe copies. |