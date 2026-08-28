/**
 * @id        6
 * @name      Register Tiling
 * @benefit   Reduces shared-memory traffic and bank pressure by holding a small
 *            output micro-tile in registers, which are faster than SRAM.
 * @strategy  Have each thread compute several output elements instead of one, so
 *            each value loaded from shared memory is reused across them.
 * @algorithm Square matrix multiply C = A x B at 1024x1024 with 16x16 shared tiles.
 *            Same problem as optimization 5, one level further down the hierarchy.
 * @before    One output element per thread, in a 16x16 block; every k step reads
 *            one element of each operand from shared memory.
 * @after     A 2x2 output micro-tile per thread, in an 8x8 block; each k step reads
 *            2 values per operand and reuses them for 4 multiply-accumulates.
 * @kernel_before register_tiling_before_kernel
 * @kernel_after  register_tiling_after_kernel
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | ShmemLdWaves | down | Shared-memory load wavefronts — register reuse halves shared reads per output element
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum | ShmemStWaves | any | Shared-memory store wavefronts — the same tile bytes are staged either way, by fewer threads
 * @metric l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum | ShmemLdConflicts | any | Shared load bank conflicts — wider per-thread accesses can introduce conflicts that offset the wavefront saving
 * @metric smsp__inst_executed.sum | Instructions | down | Total instructions — register accumulation replaces repeated shared-memory loads
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — register operands have far lower latency than shared memory
 */

#include "common.cuh"

using namespace opt;

constexpr int REG_TILE_DIM = 16;
constexpr int WIDTH        = 1024;
constexpr int MATMUL_NITER = 20;

// BEFORE: one output element per thread, every k step hits shared memory.
__global__ void register_tiling_before_kernel(const float* A, const float* B,
                                              float* C, int width) {
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

void launch_register_tiling_before(const float* d_A, const float* d_B,
                                   float* d_C, int width) {
    dim3 block(REG_TILE_DIM, REG_TILE_DIM);
    dim3 grid(width / REG_TILE_DIM, width / REG_TILE_DIM);
    register_tiling_before_kernel<<<grid, block>>>(d_A, d_B, d_C, width);
}

// AFTER: 2x2 micro-tile per thread, accumulated entirely in registers.
__global__ void register_tiling_after_kernel(const float* A, const float* B,
                                             float* C, int width) {
    __shared__ float sA[REG_TILE_DIM][REG_TILE_DIM];
    __shared__ float sB[REG_TILE_DIM][REG_TILE_DIM];

    int ty  = threadIdx.y;
    int tx  = threadIdx.x;
    int row = blockIdx.y * REG_TILE_DIM + ty * 2;
    int col = blockIdx.x * REG_TILE_DIM + tx * 2;

    float c_reg[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int ph = 0; ph < width / REG_TILE_DIM; ++ph) {
        // Each thread stages a 2x2 patch of both tiles.
        sA[ty * 2 + 0][tx * 2 + 0] = A[(row + 0) * width + ph * REG_TILE_DIM + tx * 2 + 0];
        sA[ty * 2 + 0][tx * 2 + 1] = A[(row + 0) * width + ph * REG_TILE_DIM + tx * 2 + 1];
        sA[ty * 2 + 1][tx * 2 + 0] = A[(row + 1) * width + ph * REG_TILE_DIM + tx * 2 + 0];
        sA[ty * 2 + 1][tx * 2 + 1] = A[(row + 1) * width + ph * REG_TILE_DIM + tx * 2 + 1];

        sB[ty * 2 + 0][tx * 2 + 0] = B[(ph * REG_TILE_DIM + ty * 2 + 0) * width + col + 0];
        sB[ty * 2 + 0][tx * 2 + 1] = B[(ph * REG_TILE_DIM + ty * 2 + 0) * width + col + 1];
        sB[ty * 2 + 1][tx * 2 + 0] = B[(ph * REG_TILE_DIM + ty * 2 + 1) * width + col + 0];
        sB[ty * 2 + 1][tx * 2 + 1] = B[(ph * REG_TILE_DIM + ty * 2 + 1) * width + col + 1];
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

    C[(row + 0) * width + col + 0] = c_reg[0][0];
    C[(row + 0) * width + col + 1] = c_reg[0][1];
    C[(row + 1) * width + col + 0] = c_reg[1][0];
    C[(row + 1) * width + col + 1] = c_reg[1][1];
}

void launch_register_tiling_after(const float* d_A, const float* d_B,
                                  float* d_C, int width) {
    dim3 block(REG_TILE_DIM / 2, REG_TILE_DIM / 2);
    dim3 grid(width / REG_TILE_DIM, width / REG_TILE_DIM);
    register_tiling_after_kernel<<<grid, block>>>(d_A, d_B, d_C, width);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    const size_t elems = static_cast<size_t>(WIDTH) * WIDTH;
    DeviceArray<float> d_A(elems), d_B(elems), d_C(elems);
    d_A.upload(patternFloats(elems));
    d_B.upload(patternFloats(elems));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_C.zero();
        launch_register_tiling_before(d_A, d_B, d_C, WIDTH);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_C.download();

        d_C.zero();
        launch_register_tiling_after(d_A, d_B, d_C, WIDTH);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allClose(d_C.download(), ref, 1e-3f, 1e-2f);
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(MATMUL_NITER, [&] { launch_register_tiling_before(d_A, d_B, d_C, WIDTH); });
    if (wantsAfter(mode))
        ta = timeKernelMs(MATMUL_NITER, [&] { launch_register_tiling_after(d_A, d_B, d_C, WIDTH); });

    return report(mode, "6. Register Tiling", tb, ta, verified);
}
