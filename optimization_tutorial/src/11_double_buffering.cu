/**
 * @id        11
 * @name      Double Buffering
 * @benefit   Removes one barrier per tile by eliminating the write-after-read hazard
 *            between computing a tile and loading the next one.
 * @strategy  Ping-pong between two shared-memory buffers, so the next tile is staged
 *            into one buffer while the current tile is still being read from the other.
 * @algorithm Element-wise transform out[i] = in[i]^2 + 3 applied tile by tile over
 *            1M floats, 128 elements per tile, in a single 128-thread block.
 * @before    One buffer, so each iteration needs two barriers: one after the load and
 *            one after the compute, to stop the next load overwriting live data.
 * @after     Two buffers with roles swapped each iteration, so a single barrier per
 *            tile suffices.
 * @kernel_before double_buffering_before_kernel
 * @kernel_after  double_buffering_after_kernel
 *
 * A single block cannot fill the GPU, so this measures barrier overhead rather
 * than throughput; occupancy stays low in both variants by construction.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric smsp__inst_executed.sum | Instructions | down | Total instructions — one fewer barrier per tile across 8192 tiles
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — fewer barriers means less time with all warps blocked
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | ShmemLdWaves | any | Shared load wavefronts — identical data volume confirms the two variants are equivalent
 * @metric smsp__warps_active.avg.pct_of_peak_sustained_active | Occupancy% | any | Achieved occupancy — single-block kernel, so this stays low by design
 */

#include "common.cuh"

using namespace opt;

constexpr int DBUF_TILE = 128;

// BEFORE: one buffer forces a second barrier to protect against overwrite.
__global__ void double_buffering_before_kernel(const float* in, float* out, int numTiles) {
    __shared__ float sBuf[DBUF_TILE];
    int tid = threadIdx.x;

    for (int ph = 0; ph < numTiles; ++ph) {
        sBuf[tid] = in[ph * DBUF_TILE + tid];
        __syncthreads();

        out[ph * DBUF_TILE + tid] = dummyProcess(sBuf[tid]);
        __syncthreads();
    }
}

void launch_double_buffering_before(const float* d_in, float* d_out, int numTiles) {
    double_buffering_before_kernel<<<1, DBUF_TILE>>>(d_in, d_out, numTiles);
}

// AFTER: two buffers, so staging the next tile cannot clobber the live one.
__global__ void double_buffering_after_kernel(const float* in, float* out, int numTiles) {
    __shared__ float sBuf[2][DBUF_TILE];
    int tid = threadIdx.x;
    int readIdx = 0, writeIdx = 1;

    sBuf[0][tid] = in[tid];
    __syncthreads();

    for (int ph = 1; ph < numTiles; ++ph) {
        sBuf[writeIdx][tid] = in[ph * DBUF_TILE + tid];
        out[(ph - 1) * DBUF_TILE + tid] = dummyProcess(sBuf[readIdx][tid]);

        __syncthreads();
        readIdx  ^= 1;
        writeIdx ^= 1;
    }
    out[(numTiles - 1) * DBUF_TILE + tid] = dummyProcess(sBuf[readIdx][tid]);
}

void launch_double_buffering_after(const float* d_in, float* d_out, int numTiles) {
    double_buffering_after_kernel<<<1, DBUF_TILE>>>(d_in, d_out, numTiles);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    const int numTiles = N / DBUF_TILE;
    DeviceArray<float> d_in(N), d_out(N);
    d_in.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_out.zero();
        launch_double_buffering_before(d_in, d_out, numTiles);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_out.download();

        d_out.zero();
        launch_double_buffering_after(d_in, d_out, numTiles);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_out.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_double_buffering_before(d_in, d_out, numTiles); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_double_buffering_after(d_in, d_out, numTiles); });

    return report(mode, "11. Double Buffering", tb, ta, verified);
}
