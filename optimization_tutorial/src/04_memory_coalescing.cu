/**
 * @id        4
 * @name      Memory Coalescing
 * @benefit   Maximizes DRAM burst efficiency and cache-line utilization; a warp
 *            fetches one 128-byte segment instead of 32 scattered ones.
 * @strategy  Map consecutive threadIdx.x values onto consecutive addresses so the
 *            memory system can merge a warp's 32 accesses into a single transaction.
 * @algorithm Array copy out[k] = in[k] over a 1024x1024 grid. Both kernels write
 *            exactly the same values to exactly the same locations; only the
 *            mapping from thread to index differs.
 * @before    Thread (x,y) owns column-major index x*height + y, so consecutive
 *            lanes in a warp touch addresses `height` floats apart.
 * @after     Thread (x,y) owns row-major index y*width + x, so consecutive lanes
 *            touch adjacent floats and the warp's accesses coalesce.
 * @kernel_before memory_coalescing_before_kernel
 * @kernel_after  memory_coalescing_after_kernel
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum | GlobalLdSectors | down | 32-byte sectors fetched — the uncoalesced version pulls a separate sector per lane
 * @metric l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum | GlobalLdReqs | any | Global load requests — the sectors/requests ratio is the coalescing inefficiency factor
 * @metric lts__t_sectors_op_read.sum | L2Read(sect) | down | L2 read sectors — strided access drags in lines whose bytes are mostly discarded
 * @metric ?dram__bytes_read.sum | DRAMRead(B) | down | Bytes fetched from device memory — shows whether the wasted sectors reached memory or were absorbed by L2
 * @metric gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed | Memory% | up | Memory pipeline utilization — coalesced bursts use a far larger share of each transfer
 * @metric l1tex__t_sector_hit_rate.pct | L1hit% | up | L1 sector hit rate — adjacent lanes reuse the same cache lines
 */

#include "common.cuh"

using namespace opt;

constexpr int WIDTH  = 1024;
constexpr int HEIGHT = N / 1024;

// BEFORE: warp stride is `height`, so each lane needs its own sector.
__global__ void memory_coalescing_before_kernel(const float* in, float* out,
                                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = x * height + y;
        out[idx] = in[idx];
    }
}

void launch_memory_coalescing_before(const float* d_in, float* d_out,
                                     int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    memory_coalescing_before_kernel<<<grid, block>>>(d_in, d_out, width, height);
}

// AFTER: warp stride is 1, so 32 lanes merge into one transaction.
__global__ void memory_coalescing_after_kernel(const float* in, float* out,
                                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        out[idx] = in[idx];
    }
}

void launch_memory_coalescing_after(const float* d_in, float* d_out,
                                    int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    memory_coalescing_after_kernel<<<grid, block>>>(d_in, d_out, width, height);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_in(N), d_out(N);
    d_in.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_out.zero();
        launch_memory_coalescing_before(d_in, d_out, WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_out.download();

        d_out.zero();
        launch_memory_coalescing_after(d_in, d_out, WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_out.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_memory_coalescing_before(d_in, d_out, WIDTH, HEIGHT); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_memory_coalescing_after(d_in, d_out, WIDTH, HEIGHT); });

    return report(mode, "4. Memory Coalescing", tb, ta, verified);
}
