/**
 * CUDA Optimizations Checklist - Reference Implementations
 * 
 * Contains complete before & after kernels and host launch functions for all
 * optimizations in the checklist.
 * 
 * Compilation:
 *   nvcc -O3 -arch=native cuda_optimizations_all.cu -o cuda_optimizations
 */

#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Helper dummy device arithmetic
__device__ inline float dummyTaskA(float val) { return val * 2.0f + 1.0f; }
__device__ inline float dummyTaskB(float val) { return val * 0.5f - 1.0f; }
__device__ inline float dummyProcess(float val) { return val * val + 3.0f; }

// ============================================================================
// 1. OCCUPANCY TUNING
// ============================================================================
/**
 * Optimization: Occupancy Tuning
 * Benefit: Increases active warps to hide memory/pipeline latencies.
 * Strategy: Avoid suboptimal arbitrary block sizes; leverage the CUDA 
 *           Occupancy API to determine the optimal block size for the SM.
 * Algorithm: Element-wise vector addition c[i] = a[i] + b[i] over N floats.
 */

// Simple vector add kernel
__global__ void occupancy_vectorAdd_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// BEFORE: Arbitrary, low thread-count per block causing low warp occupancy
void launch_occupancy_tuning_before(const float* d_a, const float* d_b, float* d_c, int n, cudaStream_t stream = 0) {
    dim3 blockDim(32); // 1 warp per block: severe SM occupancy penalty
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    occupancy_vectorAdd_kernel<<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, n);
}

// AFTER: Occupancy calculator query dynamically sizes thread blocks
void launch_occupancy_tuning_after(const float* d_a, const float* d_b, float* d_c, int n, cudaStream_t stream = 0) {
    int minGridSize = 0, blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, occupancy_vectorAdd_kernel, 0, 0));
    std::cout << "  [Occupancy Tuning] optimal blockSize=" << blockSize << "\n";
    dim3 gridDim((n + blockSize - 1) / blockSize);
    occupancy_vectorAdd_kernel<<<gridDim, blockSize, 0, stream>>>(d_a, d_b, d_c, n);
}

// ============================================================================
// 2. LOOP UNROLLING & PROMOTING LOCAL ARRAYS TO REGISTERS
// ============================================================================
/**
 * Optimization: Loop Unrolling
 * Benefit: Reduces branch instructions and instruction stalls; enables compiler
 *           to promote dynamically indexed local arrays to physical registers.
 * Strategy: Use #pragma unroll with constant iteration bounds.
 * Algorithm: Element-wise scale out[i] = in[i] * 2.0f, processing 4 elements per thread.
 */

// BEFORE: Dynamic loop indexing forces the local array to reside in local memory (DRAM stack)
__global__ void loop_unrolling_before_kernel(const float* in, float* out, int numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 4 + 3 < numElements) {
        float localArr[4];
        #pragma nounroll
        for (int i = 0; i < 4; ++i) {
            localArr[i] = in[tid * 4 + i];
            out[tid * 4 + i] = localArr[i] * 2.0f;
        }
    }
}

void launch_loop_unrolling_before(const float* d_in, float* d_out, int numElements, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = ((numElements / 4) + threads - 1) / threads;
    loop_unrolling_before_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, numElements);
}

// AFTER: Explicit unrolling maps localArr directly into register space
__global__ void loop_unrolling_after_kernel(const float* in, float* out, int numElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 4 + 3 < numElements) {
        float localArr[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            localArr[i] = in[tid * 4 + i];
            out[tid * 4 + i] = localArr[i] * 2.0f;
        }
    }
}

void launch_loop_unrolling_after(const float* d_in, float* d_out, int numElements, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = ((numElements / 4) + threads - 1) / threads;
    loop_unrolling_after_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, numElements);
}

// ============================================================================
// 3. REDUCING CONTROL DIVERGENCE
// ============================================================================
/**
 * Optimization: Reducing Control Divergence
 * Benefit: Prevents warp lane serialization and maximizes SIMD execution units.
 * Strategy: Eliminate the branch entirely via branchless arithmetic — compute both
 *           results and select with 0/1 weights; no warp serialization possible.
 * Algorithm: Per-element dispatch — even threads apply dummyTaskA(x)=2x+1, odd threads apply dummyTaskB(x)=0.5x-1.
 */

// BEFORE: Threads alternate branches based on parity (100% intra-warp branch divergence)
__global__ void control_divergence_before_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (threadIdx.x % 2 == 0) {
            out[idx] = dummyTaskA(in[idx]);
        } else {
            out[idx] = dummyTaskB(in[idx]);
        }
    }
}

void launch_control_divergence_before(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    control_divergence_before_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

// AFTER: Branchless arithmetic selection; both tasks computed, selected by weight
__global__ void control_divergence_after_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // task=1 for even threads → TaskA; task=0 for odd → TaskB
        int task = 1 - (threadIdx.x % 2);
        out[idx] = dummyTaskA(in[idx]) * task + dummyTaskB(in[idx]) * (1 - task);
    }
}

void launch_control_divergence_after(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    control_divergence_after_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

// ============================================================================
// 4. COALESCEABLE GLOBAL MEMORY ACCESSES
// ============================================================================
/**
 * Optimization: Coalesced Global Memory Access
 * Benefit: Maximize DRAM burst efficiency and cache-line utilization (128-byte segments).
 * Strategy: Map contiguous threadIdx.x values to contiguous memory addresses so that
 *           a whole warp issues a single merged 128-byte transaction instead of
 *           width separate serialized transactions.
 * Algorithm: Array copy out[k] = in[k] over a width×height grid.
 *           Both kernels compute out[k] = in[k] for every element k, producing
 *           identical output. The difference is purely in which thread owns which k:
 *           - Before: thread (x,y) owns index x*height+y  → warp stride = height (uncoalesced)
 *           - After:  thread (x,y) owns index y*width+x   → warp stride = 1     (coalesced)
 */

// BEFORE: thread (x,y) owns column-major index x*height+y; consecutive warp threads stride by 'height'
__global__ void memory_coalescing_before_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        // Consecutive threadIdx.x values access addresses height apart → uncoalesced
        int idx = x * height + y;
        out[idx] = in[idx];

    }
}

void launch_memory_coalescing_before(const float* d_in, float* d_out, int width, int height, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    memory_coalescing_before_kernel<<<grid, block, 0, stream>>>(d_in, d_out, width, height);
}

// AFTER: thread (x,y) owns row-major index y*width+x; consecutive warp threads are adjacent in memory
__global__ void memory_coalescing_after_kernel(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        out[idx] = in[idx];
    }
}

void launch_memory_coalescing_after(const float* d_in, float* d_out, int width, int height, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    memory_coalescing_after_kernel<<<grid, block, 0, stream>>>(d_in, d_out, width, height);
}

// ============================================================================
// 5. SHARED MEMORY TILING
// ============================================================================
/**
 * Optimization: Shared Memory Tiling
 * Benefit: Reduces repeated global memory reads by caching shared data in SM SRAM.
 * Strategy: Cooperatively load a TILE_WIDTH×TILE_WIDTH block of M and N into __shared__; each
 *           thread accumulates its partial dot product from SRAM, not DRAM.
 * Algorithm: Square matrix multiplication P = M × N (Width×Width).
 *           Example taken from Section 5.4 of "Programming Massively Parallel
 *           Processors" (Kirk & Hwu), 5th edition
 */

#define TILE_WIDTH 32

// BEFORE: Every thread reads M and N from global memory on every k iteration
__global__ void shared_tiling_matmul_before(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width && col < Width) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

void launch_shared_memory_tiling_before(const float* d_M, const float* d_N, float* d_P, int width, cudaStream_t stream = 0) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
    shared_tiling_matmul_before<<<grid, block, 0, stream>>>(d_M, d_N, d_P, width);
}

// AFTER: Cooperatively stage TILE_WIDTH×TILE_WIDTH tiles into shared memory; Pvalue lives in a register
__global__ void shared_tiling_matmul_after(const float* M, const float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0f; // accumulates in register across all phases

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < Width && ph * TILE_WIDTH + threadIdx.x < Width)
            Mds[threadIdx.y][threadIdx.x] = M[Row * Width + ph * TILE_WIDTH + threadIdx.x];
        else
            Mds[threadIdx.y][threadIdx.x] = 0.0f;

        if (Col < Width && ph * TILE_WIDTH + threadIdx.y < Width)
            Nds[threadIdx.y][threadIdx.x] = N[(ph * TILE_WIDTH + threadIdx.y) * Width + Col];
        else
            Nds[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < Width && Col < Width) P[Row * Width + Col] = Pvalue;
}

void launch_shared_memory_tiling_after(const float* d_M, const float* d_N, float* d_P, int width, cudaStream_t stream = 0) {
    // NOTE(cj): The number of threads in a block matches the number size of the TILE itself
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
    shared_tiling_matmul_after<<<grid, block, 0, stream>>>(d_M, d_N, d_P, width);
}

// ============================================================================
// 6. REGISTER TILING
// ============================================================================
/**
 * Optimization: Register Tiling
 * Benefit: Reduces shared memory bank/bandwidth stalls by caching micro-tiles in registers.
 * Strategy: Compute multiple output elements per thread inside register arrays.
 * Algorithm: Square matrix multiplication C = A × B (width×width); same as opt 5 but with 2×2 per-thread register micro-tiles.
 */

#define REG_TILE_DIM 16

// BEFORE: Accumulates directly with continuous shared memory lookups
__global__ void register_tiling_before_kernel(const float* A, const float* B, float* C, int width) {
    __shared__ float sA[REG_TILE_DIM][REG_TILE_DIM];
    __shared__ float sB[REG_TILE_DIM][REG_TILE_DIM];

    int row = blockIdx.y * REG_TILE_DIM + threadIdx.y;
    int col = blockIdx.x * REG_TILE_DIM + threadIdx.x;
    float pValue = 0.0f;

    for (int ph = 0; ph < width / REG_TILE_DIM; ++ph) {
        sA[threadIdx.y][threadIdx.x] = A[row * width + ph * REG_TILE_DIM + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(ph * REG_TILE_DIM + threadIdx.y) * width + col];
        __syncthreads();

        for (int k = 0; k < REG_TILE_DIM; ++k) {
            pValue += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < width && col < width) C[row * width + col] = pValue;
}

void launch_register_tiling_before(const float* d_A, const float* d_B, float* d_C, int width, cudaStream_t stream = 0) {
    dim3 block(REG_TILE_DIM, REG_TILE_DIM);
    dim3 grid(width / REG_TILE_DIM, width / REG_TILE_DIM);
    register_tiling_before_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, width);
}

// AFTER: 2x2 micro-tile accumulation per thread entirely inside local registers
__global__ void register_tiling_after_kernel(const float* A, const float* B, float* C, int width) {
    __shared__ float sA[REG_TILE_DIM][REG_TILE_DIM];
    __shared__ float sB[REG_TILE_DIM][REG_TILE_DIM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = (blockIdx.y * REG_TILE_DIM + ty * 2);
    int col = (blockIdx.x * REG_TILE_DIM + tx * 2);

    float c_reg[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int ph = 0; ph < width / REG_TILE_DIM; ++ph) {
        // Cooperatively load 2x2 elements per thread into shared memory
        sA[ty * 2 + 0][tx * 2 + 0] = A[(row + 0) * width + ph * REG_TILE_DIM + (tx * 2 + 0)];
        sA[ty * 2 + 0][tx * 2 + 1] = A[(row + 0) * width + ph * REG_TILE_DIM + (tx * 2 + 1)];
        sA[ty * 2 + 1][tx * 2 + 0] = A[(row + 1) * width + ph * REG_TILE_DIM + (tx * 2 + 0)];
        sA[ty * 2 + 1][tx * 2 + 1] = A[(row + 1) * width + ph * REG_TILE_DIM + (tx * 2 + 1)];

        sB[ty * 2 + 0][tx * 2 + 0] = B[(ph * REG_TILE_DIM + ty * 2 + 0) * width + (col + 0)];
        sB[ty * 2 + 0][tx * 2 + 1] = B[(ph * REG_TILE_DIM + ty * 2 + 0) * width + (col + 1)];
        sB[ty * 2 + 1][tx * 2 + 0] = B[(ph * REG_TILE_DIM + ty * 2 + 1) * width + (col + 0)];
        sB[ty * 2 + 1][tx * 2 + 1] = B[(ph * REG_TILE_DIM + ty * 2 + 1) * width + (col + 1)];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < REG_TILE_DIM; ++k) {
            float a0 = sA[ty * 2 + 0][k];
            float a1 = sA[ty * 2 + 1][k];
            float b0 = sB[k][tx * 2 + 0];
            float b1 = sB[k][tx * 2 + 1];

            c_reg[0][0] += a0 * b0;
            c_reg[0][1] += a0 * b1;
            c_reg[1][0] += a1 * b0;
            c_reg[1][1] += a1 * b1;
        }
        __syncthreads();
    }

    C[(row + 0) * width + (col + 0)] = c_reg[0][0];
    C[(row + 0) * width + (col + 1)] = c_reg[0][1];
    C[(row + 1) * width + (col + 0)] = c_reg[1][0];
    C[(row + 1) * width + (col + 1)] = c_reg[1][1];
}

void launch_register_tiling_after(const float* d_A, const float* d_B, float* d_C, int width, cudaStream_t stream = 0) {
    dim3 block(REG_TILE_DIM / 2, REG_TILE_DIM / 2);
    dim3 grid(width / REG_TILE_DIM, width / REG_TILE_DIM);
    register_tiling_after_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, width);
}

// ============================================================================
// 7. VECTOR LOADS AND STORES (128-BIT VECTORIZATION)
// ============================================================================
/**
 * Optimization: Vectorized Memory Transactions
 * Benefit: Emits LDG.E.128 instructions, reducing load issue count and instruction overhead.
 * Strategy: Reinterpret contiguous pointers as float4/uint4 vector types.
 * Algorithm: Element-wise scale out[i] = in[i] * 2.0f over N floats.
 */

// BEFORE: Scalar 32-bit (4-byte) single-element transfers
__global__ void vector_loads_before_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

void launch_vector_loads_before(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_loads_before_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

// AFTER: Vectorized 128-bit (16-byte) float4 transfers
__global__ void vector_loads_after_kernel(const float4* in, float4* out, int nVec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nVec) {
        float4 val = in[idx];
        val.x *= 2.0f; val.y *= 2.0f; val.z *= 2.0f; val.w *= 2.0f;
        out[idx] = val;
    }
}

void launch_vector_loads_after(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int nVec = n / 4;
    int threads = 256;
    int blocks = (nVec + threads - 1) / threads;
    vector_loads_after_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float4*>(d_in),
        reinterpret_cast<float4*>(d_out),
        nVec
    );
}

// ============================================================================
// 8. AVOIDING SHARED MEMORY BANK CONFLICTS
// ============================================================================
/**
 * Optimization: Shared Memory Bank Conflict Avoidance
 * Benefit: Eliminates serialized access across the 32 shared memory banks.
 * Strategy: Pad 2D shared memory array dimensions by +1 column.
 * Algorithm: In-place 32×32 matrix transpose through shared memory: out[x*32+y] = sData[y][x] = x+y.
 */

// BEFORE: 32 threads simultaneously accessing column 0 of a 32-column structure (32-way conflict)
__global__ void bank_conflicts_before_kernel(float* out) {
    __shared__ float sData[32][32];
    sData[threadIdx.x][threadIdx.y] = threadIdx.x + threadIdx.y;
    __syncthreads();

    // Stride-32 column access: threadIdx.x addresses all land in bank 0
    out[threadIdx.x * 32 + threadIdx.y] = sData[threadIdx.y][threadIdx.x];
}

void launch_bank_conflicts_before(float* d_out, cudaStream_t stream = 0) {
    dim3 block(32, 32);
    bank_conflicts_before_kernel<<<1, block, 0, stream>>>(d_out);
}

// AFTER: +1 padding skews row alignments, mapping column accesses to distinct banks
__global__ void bank_conflicts_after_kernel(float* out) {
    __shared__ float sData[32][33]; // Padded stride
    sData[threadIdx.x][threadIdx.y] = threadIdx.x + threadIdx.y;
    __syncthreads();

    // Successive threads hit successive banks (bank = (row * 33) % 32)
    out[threadIdx.x * 32 + threadIdx.y] = sData[threadIdx.y][threadIdx.x];
}

void launch_bank_conflicts_after(float* d_out, cudaStream_t stream = 0) {
    dim3 block(32, 32);
    bank_conflicts_after_kernel<<<1, block, 0, stream>>>(d_out);
}

// ============================================================================
// 9. PRIVATIZATION
// ============================================================================
/**
 * Optimization: Privatization
 * Benefit: Decreases atomic lock contention and DRAM traffic on shared hot spots.
 * Strategy: Accumulate into block-private shared memory, commit once to global memory.
 * Algorithm: 256-bin histogram of N integer values.
 */

#define NUM_BINS 256

// BEFORE: Grid-wide atomic contention on central DRAM bins
__global__ void privatization_before_kernel(const int* data, int* globalHist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&globalHist[data[i]], 1);
    }
}

void launch_privatization_before(const int* d_data, int* d_globalHist, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = 64;
    privatization_before_kernel<<<blocks, threads, 0, stream>>>(d_data, d_globalHist, n);
}

// AFTER: Thread blocks aggregate privately in __shared__ SRAM; single flush to DRAM
__global__ void privatization_after_kernel(const int* data, int* globalHist, int n) {
    __shared__ int localHist[NUM_BINS];

    if (threadIdx.x < NUM_BINS) {
        localHist[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&localHist[data[i]], 1);
    }
    __syncthreads();

    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&globalHist[threadIdx.x], localHist[threadIdx.x]);
    }
}

void launch_privatization_after(const int* d_data, int* d_globalHist, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = 64;
    privatization_after_kernel<<<blocks, threads, 0, stream>>>(d_data, d_globalHist, n);
}

// ============================================================================
// 10. WARP-LEVEL PRIMITIVES
// ============================================================================
/**
 * Optimization: Warp-Level Primitives
 * Benefit: Removes block-wide synchronization barriers and shared memory traffic.
 * Strategy: Use warp shuffle intrinsics (__shfl_down_sync) to reduce in register space.
 * Algorithm: Reduction sum of N floats to a single scalar.
 */

// BEFORE: Block reduction staged in shared memory with multiple __syncthreads() barriers
__global__ void warp_primitives_before_kernel(const float* in, float* out) {
    __shared__ float sData[256];
    int tid = threadIdx.x;
    sData[tid] = in[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sData[tid] += sData[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out, sData[0]);
    }
}

void launch_warp_primitives_before(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = n / threads;
    warp_primitives_before_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out);
}

// AFTER: Register-level warp shuffle reduction with no shared memory or barriers
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_primitives_after_kernel(const float* in, float* out) {
    __shared__ float warpSums[8]; // Max 8 warps in 256-thread block
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;

    float val = in[blockIdx.x * blockDim.x + tid];
    val = warpReduceSum(val);

    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    // Final warp aggregates partial warp sums
    if (warpId == 0) {
        float sum = (tid < (blockDim.x / 32)) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (tid == 0) atomicAdd(out, sum);
    }
}

void launch_warp_primitives_after(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = n / threads;
    warp_primitives_after_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out);
}

// ============================================================================
// 11. DOUBLE BUFFERING
// ============================================================================
/**
 * Optimization: Double Buffering
 * Benefit: Eliminates Write-After-Read false dependencies and removes one barrier per loop.
 * Strategy: Ping-pong across two shared memory buffers.
 * Algorithm: Element-wise transform out[i] = dummyProcess(in[i]) = in[i]²+3 applied tile-by-tile (DBUF_TILE=128 elements per tile).
 */

#define DBUF_TILE 128

// BEFORE: Two synchronization barriers needed per phase to prevent RAW/WAR hazards
__global__ void double_buffering_before_kernel(const float* in, float* out, int numTiles) {
    __shared__ float sBuf[DBUF_TILE];
    int tid = threadIdx.x;

    for (int ph = 0; ph < numTiles; ++ph) {
        sBuf[tid] = in[ph * DBUF_TILE + tid];
        __syncthreads(); // Wait for buffer load

        out[ph * DBUF_TILE + tid] = dummyProcess(sBuf[tid]);
        __syncthreads(); // Wait for compute to finish before next overwrite
    }
}

void launch_double_buffering_before(const float* d_in, float* d_out, int numTiles, cudaStream_t stream = 0) {
    double_buffering_before_kernel<<<1, DBUF_TILE, 0, stream>>>(d_in, d_out, numTiles);
}

// AFTER: Asynchronous write buffer loading overlapping read buffer computation
__global__ void double_buffering_after_kernel(const float* in, float* out, int numTiles) {
    __shared__ float sBuf[2][DBUF_TILE];
    int tid = threadIdx.x;
    int readIdx = 0, writeIdx = 1;

    // Prefetch first tile
    sBuf[0][tid] = in[tid];
    __syncthreads();

    for (int ph = 1; ph < numTiles; ++ph) {
        // Stage next tile in separate buffer
        sBuf[writeIdx][tid] = in[ph * DBUF_TILE + tid];

        // Process current buffer concurrently
        out[(ph - 1) * DBUF_TILE + tid] = dummyProcess(sBuf[readIdx][tid]);

        __syncthreads(); // Single barrier protecting both buffers
        readIdx ^= 1;
        writeIdx ^= 1;
    }
    // Flush trailing tile
    out[(numTiles - 1) * DBUF_TILE + tid] = dummyProcess(sBuf[readIdx][tid]);
}

void launch_double_buffering_after(const float* d_in, float* d_out, int numTiles, cudaStream_t stream = 0) {
    double_buffering_after_kernel<<<1, DBUF_TILE, 0, stream>>>(d_in, d_out, numTiles);
}

// ============================================================================
// 12. THREAD COARSENING
// ============================================================================
/**
 * Optimization: Thread Coarsening
 * Benefit: Amortizes indexing, synchronization, and kernel launch overhead.
 * Strategy: Assign multiple serial workload elements per individual CUDA thread.
 * Algorithm: Element-wise scale out[i] = in[i] * 2.0f over N floats, with 4 elements per thread.
 */

// BEFORE: Single element per thread
__global__ void thread_coarsening_before_kernel(const float* in, float* out, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

void launch_thread_coarsening_before(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    thread_coarsening_before_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

// AFTER: 4 elements per thread with unrolled work loop
__global__ void thread_coarsening_after_kernel(const float* in, float* out, int n) {
    unsigned int baseIdx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    #pragma unroll
    for (unsigned int c = 0; c < 4; ++c) {
        unsigned int idx = baseIdx + c;
        if (idx < n) {
            out[idx] = in[idx] * 2.0f;
        }
    }
}

void launch_thread_coarsening_after(const float* d_in, float* d_out, int n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = ((n / 4) + threads - 1) / threads;
    thread_coarsening_after_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

// ============================================================================
// TIMING HELPER
// ============================================================================
// Warms up once, then times nIter executions with CUDA events; returns avg ms.
static float timeKernelMs(int nIter, const std::function<void()>& fn) {
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < nIter; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / nIter;
}

// ============================================================================
// MAIN TIMING HARNESS
// ============================================================================
int main() {
    std::cout << "CUDA Optimizations Timing Harness\n";

    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "Device: " << prop.name << "\n\n";

    const int N     = 1 << 20; // 1M elements
    const int NITER = 100;
    const size_t bytes = N * sizeof(float);

    float *d_in, *d_out, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_b,   bytes));
    CUDA_CHECK(cudaMalloc(&d_c,   bytes));
    CUDA_CHECK(cudaMemset(d_in, 0, bytes));
    CUDA_CHECK(cudaMemset(d_b,  0, bytes));

    float *d_scalar;
    CUDA_CHECK(cudaMalloc(&d_scalar, sizeof(float)));

    int *d_idata, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_idata, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist,  NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_idata, 0, N * sizeof(int)));

    auto printHeader = []() {
        std::cout << std::left
                  << std::setw(32) << "Optimization"
                  << std::setw(14) << "Before (ms)"
                  << std::setw(14) << "After (ms)"
                  << "Speedup\n"
                  << std::string(66, '-') << "\n";
    };

    auto printRow = [](const char* name, float before, float after) {
        std::cout << std::left  << std::setw(32) << name
                  << std::right << std::fixed
                  << std::setprecision(4) << std::setw(14) << before
                  << std::setprecision(4) << std::setw(14) << after
                  << std::setprecision(2) << std::setw(7)  << (before / after) << "x\n";
    };

    printHeader();

    // 1. Occupancy Tuning
    {
        float tb = timeKernelMs(NITER, [&]{ launch_occupancy_tuning_before(d_in, d_b, d_c, N); });
        float ta = timeKernelMs(NITER, [&]{ launch_occupancy_tuning_after(d_in, d_b, d_c, N); });
        printRow("1. Occupancy Tuning", tb, ta);
    }

    // 2. Loop Unrolling
    {
        float tb = timeKernelMs(NITER, [&]{ launch_loop_unrolling_before(d_in, d_out, N); });
        float ta = timeKernelMs(NITER, [&]{ launch_loop_unrolling_after(d_in, d_out, N); });
        printRow("2. Loop Unrolling", tb, ta);
    }

    // 3. Control Divergence
    {
        float tb = timeKernelMs(NITER, [&]{ launch_control_divergence_before(d_in, d_out, N); });
        float ta = timeKernelMs(NITER, [&]{ launch_control_divergence_after(d_in, d_out, N); });
        printRow("3. Control Divergence", tb, ta);
    }

    // 4. Memory Coalescing (1024 x 1024 matrix)
    {
        const int W = 1024, H = N / 1024;
        float tb = timeKernelMs(NITER, [&]{ launch_memory_coalescing_before(d_in, d_out, W, H); });
        float ta = timeKernelMs(NITER, [&]{ launch_memory_coalescing_after(d_in, d_out, W, H); });
        printRow("4. Memory Coalescing", tb, ta);
    }

    // 5. Shared Memory Tiling (matrix multiply 1024×1024)
    {
        const int W = 1024;
        float tb = timeKernelMs(NITER, [&]{ launch_shared_memory_tiling_before(d_in, d_b, d_out, W); });
        float ta = timeKernelMs(NITER, [&]{ launch_shared_memory_tiling_after(d_in, d_b, d_out, W); });
        printRow("5. Shared Memory Tiling", tb, ta);
    }

    // 6. Register Tiling (1024x1024 matrix; width must be multiple of REG_TILE_DIM)
    {
        const int W = 1024;
        float tb = timeKernelMs(NITER, [&]{ launch_register_tiling_before(d_in, d_b, d_out, W); });
        float ta = timeKernelMs(NITER, [&]{ launch_register_tiling_after(d_in, d_b, d_out, W); });
        printRow("6. Register Tiling", tb, ta);
    }

    // 7. Vector Loads
    {
        float tb = timeKernelMs(NITER, [&]{ launch_vector_loads_before(d_in, d_out, N); });
        float ta = timeKernelMs(NITER, [&]{ launch_vector_loads_after(d_in, d_out, N); });
        printRow("7. Vector Loads", tb, ta);
    }

    // 8. Bank Conflicts (32x32 single-block kernel; use more iterations for stable timing)
    {
        const int NITER_BANK = 10000;
        float tb = timeKernelMs(NITER_BANK, [&]{ launch_bank_conflicts_before(d_out); });
        float ta = timeKernelMs(NITER_BANK, [&]{ launch_bank_conflicts_after(d_out); });
        printRow("8. Bank Conflicts", tb, ta);
    }

    // 9. Privatization (histogram; reset bins between before/after to avoid overflow)
    {
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
        float tb = timeKernelMs(NITER, [&]{ launch_privatization_before(d_idata, d_hist, N); });
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
        float ta = timeKernelMs(NITER, [&]{ launch_privatization_after(d_idata, d_hist, N); });
        printRow("9. Privatization", tb, ta);
    }

    // 10. Warp Primitives (scalar reduction; reset accumulator between before/after)
    {
        CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
        float tb = timeKernelMs(NITER, [&]{ launch_warp_primitives_before(d_in, d_scalar, N); });
        CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
        float ta = timeKernelMs(NITER, [&]{ launch_warp_primitives_after(d_in, d_scalar, N); });
        printRow("10. Warp Primitives", tb, ta);
    }

    // 11. Double Buffering
    {
        const int numTiles = N / DBUF_TILE;
        float tb = timeKernelMs(NITER, [&]{ launch_double_buffering_before(d_in, d_out, numTiles); });
        float ta = timeKernelMs(NITER, [&]{ launch_double_buffering_after(d_in, d_out, numTiles); });
        printRow("11. Double Buffering", tb, ta);
    }

    // 12. Thread Coarsening
    {
        float tb = timeKernelMs(NITER, [&]{ launch_thread_coarsening_before(d_in, d_out, N); });
        float ta = timeKernelMs(NITER, [&]{ launch_thread_coarsening_after(d_in, d_out, N); });
        printRow("12. Thread Coarsening", tb, ta);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_scalar));
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_hist));

    return 0;
}