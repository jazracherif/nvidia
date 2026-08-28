/**
 * @id        9
 * @name      Privatization
 * @benefit   Removes global atomic contention on a small set of hot addresses by
 *            giving each block its own private copy to accumulate into.
 * @strategy  Accumulate into a block-private __shared__ histogram, then commit one
 *            atomic per bin per block to the global histogram.
 * @algorithm 256-bin histogram over 1M integers. The input is deliberately skewed:
 *            a quarter of all values fall in bin 0, so the naive version serializes
 *            heavily on a single address.
 * @before    Every element issues an atomicAdd straight to the global histogram, so
 *            all blocks contend on the same DRAM-backed bins.
 * @after     Threads accumulate in shared memory, then each block flushes its 256
 *            bins with one atomic each — 64 blocks x 256 bins instead of 1M atomics.
 * @kernel_before privatization_before_kernel
 * @kernel_after  privatization_after_kernel
 *
 * The optimized version adds shared-memory atomics, so shared store traffic is
 * expected to rise while global atomic traffic collapses.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum | GlobalAtom(B) | down | Global atomic bytes — should fall by roughly the number of elements per block
 * @metric lts__t_sectors_op_read.sum | L2Read(sect) | down | L2 read sectors — global read-modify-write traffic mostly disappears
 * @metric lts__t_sectors_op_write.sum | L2Write(sect) | down | L2 write sectors — one flush per block replaces one write per element
 * @metric ?dram__bytes_read.sum | DRAMRead(B) | down | Device-memory reads — the 256 bins are tiny and L2-resident, so this may already be near zero
 * @metric ?dram__bytes_write.sum | DRAMWrite(B) | down | Device-memory writes — how much of the atomic traffic actually reached memory
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum | ShmemStWaves | up | Shared store wavefronts — confirms the contention moved into fast SRAM
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — less stalling on serialized global atomics
 */

#include "common.cuh"

using namespace opt;

constexpr int NUM_BINS = 256;
constexpr int BLOCKS   = 64;
constexpr int THREADS  = 256;

// BEFORE: every element contends on the global histogram.
__global__ void privatization_before_kernel(const int* data, int* globalHist, int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&globalHist[data[i]], 1);
    }
}

void launch_privatization_before(const int* d_data, int* d_hist, int n) {
    privatization_before_kernel<<<BLOCKS, THREADS>>>(d_data, d_hist, n);
}

// AFTER: block-private histogram in shared memory, one flush per bin per block.
__global__ void privatization_after_kernel(const int* data, int* globalHist, int n) {
    __shared__ int localHist[NUM_BINS];

    if (threadIdx.x < NUM_BINS) {
        localHist[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        atomicAdd(&localHist[data[i]], 1);
    }
    __syncthreads();

    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&globalHist[threadIdx.x], localHist[threadIdx.x]);
    }
}

void launch_privatization_after(const int* d_data, int* d_hist, int n) {
    privatization_after_kernel<<<BLOCKS, THREADS>>>(d_data, d_hist, n);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<int> d_data(N), d_hist(NUM_BINS);
    d_data.upload(patternBins(N, NUM_BINS));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_hist.zero();
        launch_privatization_before(d_data, d_hist, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<int> ref = d_hist.download();

        d_hist.zero();
        launch_privatization_after(d_data, d_hist, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_hist.download());
    }

    // Bins are not reset between timed iterations; counts overflow harmlessly and
    // resetting would time the memset rather than the kernel.
    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_privatization_before(d_data, d_hist, N); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_privatization_after(d_data, d_hist, N); });

    return report(mode, "9. Privatization", tb, ta, verified);
}
