/**
 * @id        5
 * @name      Shared Memory Tiling
 * @benefit   Cuts repeated global-memory reads by staging reused data in the SM's
 *            shared-memory SRAM instead of re-fetching it from DRAM.
 * @strategy  Cooperatively load a TILE_WIDTH x TILE_WIDTH tile of each operand into
 *            __shared__, barrier once, then serve every thread's inner loop from SRAM.
 * @algorithm Square matrix multiply P = M x N at 1024x1024 with TILE_WIDTH = 32.
 *            Adapted from section 5.4 of Kirk & Hwu, "Programming Massively
 *            Parallel Processors".
 * @before    Every thread reads one element of M and one of N from global memory
 *            on every one of the 1024 k iterations.
 * @after     Each tile is read from DRAM once per block and reused TILE_WIDTH times
 *            from shared memory, with the accumulator held in a register.
 * @kernel_before shared_tiling_matmul_before
 * @kernel_after  shared_tiling_matmul_after
 *
 * The tiled version reassociates the dot product, so results match the naive
 * version only to floating-point tolerance.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric lts__t_sectors_op_read.sum | L2Read(sect) | down | L2 read sectors — tiling should cut traffic to the memory hierarchy by roughly TILE_WIDTH
 * @metric ?dram__bytes_read.sum | DRAMRead(B) | down | Bytes actually fetched from device memory — the true reuse measure, where the GPU exposes it; integrated parts have no dram__* counters
 * @metric gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed | Memory% | any | Memory pipeline utilization — falling relative to SM throughput means work moved from DRAM to SRAM
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | ShmemLdWaves | up | Shared-memory load wavefronts — should appear only in the tiled version
 * @metric l1tex__t_sector_hit_rate.pct | L1hit% | any | L1 sector hit rate — tiling improves locality, though shared memory bypasses L1 entirely
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — the tiled kernel becomes compute bound rather than DRAM bound
 */

#include "common.cuh"

using namespace opt;

constexpr int TILE_WIDTH   = 32;
constexpr int WIDTH        = 1024;
// Naive 1024^3 matmul is slow; fewer iterations still time it reliably.
constexpr int MATMUL_NITER = 20;

// BEFORE: both operands re-read from global memory on every k step.
__global__ void shared_tiling_matmul_before(const float* M, const float* Nmat,
                                            float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width && col < Width) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * Nmat[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

void launch_shared_memory_tiling_before(const float* d_M, const float* d_N,
                                        float* d_P, int width) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH,
              (width + TILE_WIDTH - 1) / TILE_WIDTH);
    shared_tiling_matmul_before<<<grid, block>>>(d_M, d_N, d_P, width);
}

// AFTER: tiles staged in shared memory, accumulator kept in a register.
__global__ void shared_tiling_matmul_after(const float* M, const float* Nmat,
                                           float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0f;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < Width && ph * TILE_WIDTH + threadIdx.x < Width)
            Mds[threadIdx.y][threadIdx.x] = M[Row * Width + ph * TILE_WIDTH + threadIdx.x];
        else
            Mds[threadIdx.y][threadIdx.x] = 0.0f;

        if (Col < Width && ph * TILE_WIDTH + threadIdx.y < Width)
            Nds[threadIdx.y][threadIdx.x] = Nmat[(ph * TILE_WIDTH + threadIdx.y) * Width + Col];
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

void launch_shared_memory_tiling_after(const float* d_M, const float* d_N,
                                       float* d_P, int width) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH,
              (width + TILE_WIDTH - 1) / TILE_WIDTH);
    shared_tiling_matmul_after<<<grid, block>>>(d_M, d_N, d_P, width);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    const size_t elems = static_cast<size_t>(WIDTH) * WIDTH;
    DeviceArray<float> d_M(elems), d_N(elems), d_P(elems);
    d_M.upload(patternFloats(elems));
    d_N.upload(patternFloats(elems));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_P.zero();
        launch_shared_memory_tiling_before(d_M, d_N, d_P, WIDTH);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_P.download();

        d_P.zero();
        launch_shared_memory_tiling_after(d_M, d_N, d_P, WIDTH);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allClose(d_P.download(), ref, 1e-3f, 1e-2f);
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(MATMUL_NITER, [&] { launch_shared_memory_tiling_before(d_M, d_N, d_P, WIDTH); });
    if (wantsAfter(mode))
        ta = timeKernelMs(MATMUL_NITER, [&] { launch_shared_memory_tiling_after(d_M, d_N, d_P, WIDTH); });

    return report(mode, "5. Shared Memory Tiling", tb, ta, verified);
}
