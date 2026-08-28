/**
 * @id        12
 * @name      Thread Coarsening
 * @benefit   Amortizes per-thread setup — index arithmetic, bounds checks and
 *            scheduling overhead — across several elements.
 * @strategy  Give each thread a small fixed batch of elements instead of one, and
 *            unroll the batch so the addresses are computed from one base index.
 * @algorithm Element-wise scale out[i] = in[i] * 2.0f over 1M floats, 4 elements
 *            per thread in the optimized variant.
 * @before    One element per thread: 1M threads, each doing full index arithmetic
 *            and a bounds check for a single multiply.
 * @after     Four elements per thread: 256K threads, with the index computed once
 *            and the batch unrolled.
 * @kernel_before thread_coarsening_before_kernel
 * @kernel_after  thread_coarsening_after_kernel
 *
 * Occupancy is expected to fall here: the win comes from doing less work per
 * element, not from keeping more warps resident. This is the one optimization in
 * the set where lower occupancy is the intended outcome.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric smsp__inst_executed.sum | Instructions | down | Total instructions — a quarter as many threads means a quarter as much per-thread overhead
 * @metric smsp__warps_active.avg.pct_of_peak_sustained_active | Occupancy% | down | Achieved occupancy — deliberately reduced; fewer, longer-lived threads
 * @metric gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed | Memory% | any | Memory pipeline utilization — identical byte volume, so this should stay roughly flat
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — less scheduling overhead per byte moved
 */

#include "common.cuh"

using namespace opt;

constexpr int COARSEN = 4;

// BEFORE: one element per thread.
__global__ void thread_coarsening_before_kernel(const float* in, float* out, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

void launch_thread_coarsening_before(const float* d_in, float* d_out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    thread_coarsening_before_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

// AFTER: COARSEN elements per thread, batch unrolled.
__global__ void thread_coarsening_after_kernel(const float* in, float* out, int n) {
    unsigned int baseIdx = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
    #pragma unroll
    for (unsigned int c = 0; c < COARSEN; ++c) {
        unsigned int idx = baseIdx + c;
        if (idx < n) {
            out[idx] = in[idx] * 2.0f;
        }
    }
}

void launch_thread_coarsening_after(const float* d_in, float* d_out, int n) {
    int threads = 256;
    int blocks  = ((n / COARSEN) + threads - 1) / threads;
    thread_coarsening_after_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_in(N), d_out(N);
    d_in.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_out.zero();
        launch_thread_coarsening_before(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_out.download();

        d_out.zero();
        launch_thread_coarsening_after(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_out.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_thread_coarsening_before(d_in, d_out, N); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_thread_coarsening_after(d_in, d_out, N); });

    return report(mode, "12. Thread Coarsening", tb, ta, verified);
}
