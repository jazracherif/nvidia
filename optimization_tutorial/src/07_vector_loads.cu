/**
 * @id        7
 * @name      Vector Loads
 * @benefit   Emits 128-bit LDG.E.128 / STG.E.128 instructions, moving the same
 *            bytes with a quarter as many load/store instructions.
 * @strategy  Reinterpret the contiguous float arrays as float4 and have each
 *            thread process one 16-byte vector instead of one 4-byte scalar.
 * @algorithm Element-wise scale out[i] = in[i] * 2.0f over 1M floats.
 * @before    One float per thread: 1M threads, each issuing a 32-bit load and store.
 * @after     One float4 per thread: 256K threads, each issuing a 128-bit load and store.
 * @kernel_before vector_loads_before_kernel
 * @kernel_after  vector_loads_after_kernel
 *
 * Byte volume is identical in both variants; only the instruction count changes,
 * so DRAM traffic should stay flat while LSU instructions drop about 4x.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric smsp__inst_executed_pipe_lsu.sum | LSU Instr | down | Load/store pipe instructions — one LDG.E.128 replaces four LDG.E.32, so expect roughly a 4x drop
 * @metric smsp__inst_executed.sum | Instructions | down | Total instructions — fewer threads also means less index arithmetic and fewer bounds checks
 * @metric gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed | Memory% | up | Memory pipeline utilization — wider transactions keep the memory system busier
 * @metric l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum | GlobalLdSectors | any | Global load sectors — the same bytes are moved, so this should stay flat
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | any | SM utilization — relieving LSU pressure frees issue slots, but the kernel stays memory bound
 */

#include "common.cuh"

using namespace opt;

// BEFORE: scalar 32-bit transfers, one element per thread.
__global__ void vector_loads_before_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

void launch_vector_loads_before(const float* d_in, float* d_out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    vector_loads_before_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

// AFTER: 128-bit transfers, four elements per thread.
__global__ void vector_loads_after_kernel(const float4* in, float4* out, int nVec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nVec) {
        float4 v = in[idx];
        v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
        out[idx] = v;
    }
}

void launch_vector_loads_after(const float* d_in, float* d_out, int n) {
    int nVec    = n / 4;
    int threads = 256;
    int blocks  = (nVec + threads - 1) / threads;
    vector_loads_after_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(d_in),
        reinterpret_cast<float4*>(d_out),
        nVec);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_in(N), d_out(N);
    d_in.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_out.zero();
        launch_vector_loads_before(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_out.download();

        d_out.zero();
        launch_vector_loads_after(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_out.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_vector_loads_before(d_in, d_out, N); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_vector_loads_after(d_in, d_out, N); });

    return report(mode, "7. Vector Loads", tb, ta, verified);
}
