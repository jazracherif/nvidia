/**
 * @id        1
 * @name      Occupancy Tuning
 * @benefit   Increases active warps per SM to hide memory and pipeline latency.
 * @strategy  Avoid arbitrary block sizes; query cudaOccupancyMaxPotentialBlockSize
 *            to pick a block size the SM can actually saturate.
 * @algorithm Element-wise vector addition c[i] = a[i] + b[i] over 1M floats.
 * @before    32 threads per block: one warp per block, so the SM hits its block
 *            limit long before its warp limit and sits mostly idle.
 * @after     Block size chosen by the occupancy API (typically 256-1024 threads),
 *            letting each SM host many more concurrent warps.
 * @kernel_before occupancy_vectorAdd_kernel
 * @kernel_after  occupancy_vectorAdd_kernel
 *
 * Both variants launch the same kernel; only the launch configuration differs,
 * which is why each variant is profiled in a separate process.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric smsp__warps_active.avg.pct_of_peak_sustained_active | Occupancy% | up | Fraction of peak warps resident — the direct target of this optimization
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM pipeline utilization — rises as more resident warps keep execution units fed
 * @metric gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed | Memory% | up | Memory pipeline utilization — this kernel is memory bound, so better warp coverage should push the memory system harder
 */

#include "common.cuh"

using namespace opt;

__global__ void occupancy_vectorAdd_kernel(const float* a, const float* b,
                                           float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// BEFORE: one warp per block starves the SM of resident warps.
void launch_occupancy_tuning_before(const float* d_a, const float* d_b,
                                    float* d_c, int n) {
    dim3 block(32);
    dim3 grid((n + block.x - 1) / block.x);
    occupancy_vectorAdd_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
}

// AFTER: block size supplied by the occupancy calculator.
void launch_occupancy_tuning_after(const float* d_a, const float* d_b,
                                   float* d_c, int n) {
    static int blockSize = 0;
    if (blockSize == 0) {
        int minGridSize = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, occupancy_vectorAdd_kernel, 0, 0));
    }
    dim3 grid((n + blockSize - 1) / blockSize);
    occupancy_vectorAdd_kernel<<<grid, blockSize>>>(d_a, d_b, d_c, n);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_a(N), d_b(N), d_c(N);
    d_a.upload(patternFloats(N));
    d_b.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_c.zero();
        launch_occupancy_tuning_before(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_c.download();

        d_c.zero();
        launch_occupancy_tuning_after(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_c.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_occupancy_tuning_before(d_a, d_b, d_c, N); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_occupancy_tuning_after(d_a, d_b, d_c, N); });

    return report(mode, "1. Occupancy Tuning", tb, ta, verified);
}
